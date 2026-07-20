
#include <array>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <chrono>

#include <yaml-cpp/yaml.h>
//#include <sentencepiece_processor.h>
#include <pqxx/pqxx>  // libpqxx
#include <Poco/Process.h>
#include <Poco/Pipe.h>
#include <Poco/PipeStream.h>

#include "indexer.h"
#include "csv.hpp"
#include "openai.hpp"
#include "nlohmann/json.hpp"
#include "cpp-chunker/chunker/chunker.cpp"


namespace fs = std::filesystem;
using namespace csv;

class Embedder {
public:
    Embedder(const std::string& pythonScriptPath) {

        Poco::Process::Args args;
        args.push_back(pythonScriptPath);

        // Create pipes
        inPipe = std::make_unique<Poco::Pipe>();   // stdin for child (parent writes to this)
        outPipe = std::make_unique<Poco::Pipe>();  // stdout from child (parent reads from this)
        errPipe = std::make_unique<Poco::Pipe>();  // stderr from child (optional)

        // Launch child process with pipes attached
        processHandle = std::make_unique<Poco::ProcessHandle>(
            Poco::Process::launch("python3", args,
                                 inPipe.get(),   // child's stdin pipe - parent writes to it
                                 outPipe.get(),  // child's stdout pipe - parent reads from it
                                 errPipe.get())  // child's stderr pipe (optional)
        );

        inStream = std::make_unique<Poco::PipeOutputStream>(*inPipe);
        outStream = std::make_unique<Poco::PipeInputStream>(*outPipe);
        // Optional: read from child's stderr if you want
        errStream = std::make_unique<Poco::PipeInputStream>(*errPipe);
    }

    ~Embedder() {
        if (processHandle) {
            // Optionally: terminate the process gracefully before destruction
            try {
                int exitCode = processHandle->tryWait();
                if (exitCode == -1) { // process still running
                    Poco::Process::kill(*processHandle);
                }
            } catch (...) {
                // Handle errors gracefully
            }
            // unique_ptr destructor runs automatically after this function ends
        }
    }

    std::vector<float> embed(const std::string& text) {
        *inStream << text << std::endl;
        inStream->flush();

        std::string line;
        std::getline(*outStream, line);
        std::cout << "Raw Python output: '" << line << "'" << std::endl;

        try {
            nlohmann::json j = nlohmann::json::parse(line);
            return j.get<std::vector<float>>();
        } catch (const std::exception& e) {
            std::cerr << "Embedder JSON parse error: " << e.what() << "\nLine: " << line << std::endl;
            return {};
        }
    }

private:
    std::unique_ptr<Poco::Pipe> inPipe;
    std::unique_ptr<Poco::Pipe> outPipe;
    std::unique_ptr<Poco::Pipe> errPipe;

    std::unique_ptr<Poco::PipeOutputStream> inStream;
    std::unique_ptr<Poco::PipeInputStream> outStream;
    std::unique_ptr<Poco::PipeInputStream> errStream;

    std::unique_ptr<Poco::ProcessHandle> processHandle;
    
};


class TextProcessor {
public:
    // Constructor initializes DB connection and loads SentencePiece model
    TextProcessor(const std::string& db_conn_str,
                  const std::string& sp_model_path,
                  const std::string& openai_api_key,
                  const std::string& python_embedder_path)
        : conn(db_conn_str), openai_client(openai_api_key), embedder(python_embedder_path)  {
        //auto status = sp.Load(sp_model_path);
        //if (!status.ok()) {
        //    throw std::runtime_error("Failed to load SentencePiece model: " + status.ToString());
        //}
    }

    // Process plain text file - reads content, chunks, embeds, and stores
    int process_text_file(const fs::path& filepath) {
        // Check file status: skip if unchanged, delete old data if modified
        int status = check_file_status(filepath);
        if (status == 1) {
            return 0;  // Skip processing (file unchanged)
        }
        // If status == 0, continue processing (new file or modified file with old data deleted)

        std::ifstream file(filepath);
        if (!file) {
            std::cerr << "Failed to open " << filepath << std::endl;
            return 1;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        std::cout << "Processing text file: " << filepath.filename().string()
                  << " (" << content.length() << " bytes)" << std::endl;

        // Process and embed the text content
        try {
            pqxx::work W(conn);

            // Record file in embedding_files table and get file_id
            int file_id = record_file_processed(W, filepath);
            if (file_id == -1) {
                std::cerr << "Failed to record file, skipping processing" << std::endl;
                return 1;
            }

            // Process the text content: chunk, embed, and store
            // Using row_num = 0 since this is a single text document (not CSV rows)
            process_and_embed_text(content, W, file_id, 0);

            W.commit();
            std::cout << "Successfully processed file: " << filepath.filename().string() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to process file: " << e.what() << std::endl;
            return 1;
        }

        return 0;
    }

    // Process 10-Ks file, tokenize, chunk with overlap, and insert into DB
    int process_10ks_file(const fs::path& filepath) {
        std::string file_name = filepath.filename().string();
        std::cout << "Parsing 10ks: " << file_name << std::endl;

        // Check file status: skip if unchanged, delete old data if modified
        int status = check_file_status(filepath);
        if (status == 1) {
            return 0;  // Skip processing (file unchanged)
        }
        // If status == 0, continue processing (new file or modified file with old data deleted)

        try {
            pqxx::work W(conn);  // Start transaction

            // Record file in embedding_files table and get file_id
            int file_id = record_file_processed(W, filepath);
            if (file_id == -1) {
                std::cerr << "Failed to record file, skipping processing" << std::endl;
                return 1;
            }

            csv::CSVFormat format;
            format.delimiter('|').variable_columns(csv::VariableColumnPolicy::IGNORE_ROW).no_header();
            csv::CSVReader reader(filepath.string(), format);

            size_t row_num = 0;
            for (csv::CSVRow& row : reader) {
                try {
                    // Extract 10-Ks specific fields from CSV row
                    if (row.size() < 9) {
                        throw std::runtime_error("Row does not have enough columns");
                    }

                    std::string company_name = row[2].get<>();
                    std::string date = row[3].get<>();
                    std::string item_1 = row[4].get<>();
                    std::string item_1a = row[5].get<>();
                    std::string item_7a = row[6].get<>();
                    std::string price = row[8].get<>();

                    std::string text = company_name + " " + date +
                        " price: " + price + " item_1: " + item_1 +
                        " item 1a: " + item_1a + " item_7a " + item_7a;

                    std::cout << company_name << " " << row.size() << " " << date << " " << price << std::endl;

                    // Process the extracted text with file_id
                    process_and_embed_text(text, W, file_id, row_num);
                    ++row_num;
                } catch (const std::exception& e) {
                    std::cerr << "Error processing row: " << e.what() << std::endl;
                }
            }

            W.commit();
        } catch (const std::exception& e) {
            std::cerr << "CSV parse error or DB failure: " << e.what() << std::endl;
            return 1;
        }

        return 0;
    }

private:
    pqxx::connection conn;
    //sentencepiece::SentencePieceProcessor sp;
    openai::OpenAI openai_client;
    Embedder embedder;  // Persistent embedder

    // Compute simple hash of file content
    std::string compute_file_hash(const fs::path& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            return "";
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

        // Simple hash using std::hash
        std::hash<std::string> hasher;
        size_t hash_value = hasher(content);

        std::stringstream ss;
        ss << std::hex << hash_value;
        return ss.str();
    }

    // Delete old embeddings and file record for a modified file
    void delete_old_file_data(pqxx::work& W, const fs::path& filepath) {
        try {
            std::string file_path = filepath.string();
            std::string file_name = filepath.filename().string();

            // Get the file_id first
            std::string get_id_sql = "SELECT id FROM embedding_files WHERE file_path = " + W.quote(file_path);
            pqxx::result R = W.exec(get_id_sql);

            if (!R.empty()) {
                int file_id = R[0][0].as<int>();

                // Delete from embeddings table using file_id (CASCADE will handle this automatically)
                // But we'll do it explicitly for clarity
                std::string delete_embeddings = "DELETE FROM embeddings WHERE file_id = " + std::to_string(file_id);
                W.exec(delete_embeddings);

                std::cout << "Deleted old embeddings for file_id: " << file_id << std::endl;
            }

            // Delete from embedding_files table (CASCADE will delete related embeddings too)
            std::string delete_file_record = "DELETE FROM embedding_files WHERE file_path = " + W.quote(file_path);
            W.exec(delete_file_record);

            std::cout << "Deleted old data for modified file: " << file_name << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error deleting old file data: " << e.what() << std::endl;
        }
    }

    // Check if file should be skipped (already processed with same hash)
    // Returns: 0 = process file, 1 = skip file, 2 = delete old data and process
    int check_file_status(const fs::path& filepath) {
        try {
            std::string file_path = filepath.string();
            std::string current_hash = compute_file_hash(filepath);

            if (current_hash.empty()) {
                return 0;  // Can't compute hash, process file
            }

            pqxx::work W(conn);
            std::string sql = "SELECT hash FROM embedding_files WHERE file_path = " + W.quote(file_path);
            pqxx::result R = W.exec(sql);

            if (R.empty()) {
                return 0;  // File not in database, process file
            }

            std::string stored_hash = R[0][0].as<std::string>();

            if (stored_hash == current_hash) {
                std::cout << "Skipping file (unchanged): " << filepath.filename().string()
                         << " (hash: " << current_hash << ")" << std::endl;
                return 1;  // Skip file
            } else {
                std::cout << "File modified, deleting old data: " << filepath.filename().string() << std::endl;
                delete_old_file_data(W, filepath);
                W.commit();
                return 0;  // Process file after deletion
            }
        } catch (const std::exception& e) {
            std::cerr << "Error checking file status: " << e.what() << std::endl;
            return 0;  // On error, process file
        }
    }

    // Record file in embedding_files table and return the file_id
    int record_file_processed(pqxx::work& W, const fs::path& filepath) {
        try {
            std::string file_path = filepath.string();
            std::string file_name = filepath.filename().string();
            std::string hash = compute_file_hash(filepath);

            // Get file's last modification time
            auto ftime = fs::last_write_time(filepath);
            auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now()
            );
            std::time_t cftime = std::chrono::system_clock::to_time_t(sctp);

            std::stringstream time_ss;
            time_ss << std::put_time(std::localtime(&cftime), "%Y-%m-%d %H:%M:%S");
            std::string last_modified = time_ss.str();

            // Insert or update record, returning the id
            std::string sql =
                "INSERT INTO embedding_files (file_path, file_name, last_modified, hash) "
                "VALUES (" + W.quote(file_path) + ", " + W.quote(file_name) + ", "
                + W.quote(last_modified) + ", " + W.quote(hash) + ") "
                "ON CONFLICT (file_path) DO UPDATE SET "
                "file_name = EXCLUDED.file_name, "
                "last_modified = EXCLUDED.last_modified, "
                "hash = EXCLUDED.hash, "
                "updated_at = CURRENT_TIMESTAMP "
                "RETURNING id";

            pqxx::result R = W.exec(sql);
            int file_id = R[0][0].as<int>();

            std::cout << "Recorded file: " << file_name << " (id: " << file_id << ", hash: " << hash << ")" << std::endl;
            return file_id;
        } catch (const std::exception& e) {
            std::cerr << "Error recording file: " << e.what() << std::endl;
            return -1;
        }
    }

    nlohmann::json create_openai_embedding(const std::string& text) {
        nlohmann::json req = {
            {"model", "text-embedding-ada-002"},
            {"input", std::vector<std::string>{text}}
        };
        try {
            return openai_client.embedding.create(req);
        } catch (const std::exception& e) {
            std::cerr << "OpenAI API call failed: " << e.what() << std::endl;
            return nullptr;
        }
    }

    std::string vector_to_pg_array(const std::vector<float>& vec) {
        std::string s = "'[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) s += ",";
            s += std::to_string(vec[i]);
        }
        s += "]'";
        return s;
    }

    std::vector<float> json_to_vector(const nlohmann::json& response_json) {
        std::vector<float> embedding;

        try {
            if (!response_json.contains("data") || !response_json["data"].is_array() || response_json["data"].empty()) {
                throw std::runtime_error("OpenAI response missing 'data' or empty.");
            }

            const auto& embedding_json = response_json["data"][0]["embedding"];

            if (!embedding_json.is_array()) {
                throw std::runtime_error("'embedding' field is not an array.");
            }

            embedding = embedding_json.get<std::vector<float>>();
        }
        catch (const std::exception& ex) {
            std::cerr << "Failed to parse embedding from response: " << ex.what() << std::endl;
        }

        return embedding;
    }
    

    // Generic function to process text: chunk, embed, and insert into DB
    // Can be reused for any text source (10-Ks, plain text, etc.)
    void process_and_embed_text(const std::string& text, pqxx::work& W, int file_id, size_t row_num) {
        // Segment (chunk) text using semantic chunker
        std::vector<std::string> chunks = chunk_text_semantically_standalone(
            text,
            100,   // max_chunk_size
            50,    // min_chunk_size
            0.3     // min_coherence_threshold
        );

        // Process each chunk: create embedding and insert into DB
        for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
            const std::string& chunk = chunks[chunk_idx];

            //std::vector<float> embedding = embedder.embed(chunk);
            nlohmann::json emb_json = create_openai_embedding(chunk);
            std::vector<float> emb = json_to_vector(emb_json);

            // Insert into DB using new schema: file_id, row_num, chunk_idx
            std::string sql = "INSERT INTO embeddings (file_id, row_num, chunk_idx, text, emb) VALUES ("
                + std::to_string(file_id) + ", "
                + std::to_string(row_num) + ", "
                + std::to_string(chunk_idx) + ", "
                + W.quote(chunk) + ", "
                + vector_to_pg_array(emb) + ")";

            W.exec(sql);
        }
    }
};

int process_csv_file(const fs::path& filepath, pqxx::connection& C, char delimiter = ',') {

    std::string file_name = filepath.filename().string();
    std::cout << "Parsing CSV: " << file_name << std::endl;

    pqxx::work W(C);  // open transaction for each function call
    CSVFormat format;
    format.delimiter('|').variable_columns(VariableColumnPolicy::IGNORE_ROW).no_header();
    csv::CSVReader reader(filepath.string(), format);

    for (csv::CSVRow& row : reader) {
        try {

            std::string row_text;

            for (auto& field : row) {
                row_text += field.get<>() + " ";
            }

            std::string sql = "INSERT INTO embeddings_lite (source, text) VALUES (" 
                + W.quote(file_name) + ", "
                + W.quote(row_text) + ");";
            W.exec(sql);

        } catch (const std::exception& e) {
            // Handle potential errors during conversion
            std::cerr << "Error parsing row: " << e.what() << std::endl;
        }
    }
        
    return 0;
}


int run_indexer(const std::string& dir_path) {

    YAML::Node config = YAML::LoadFile("env.yaml");

    // Initialize OpenAI client
    std::string api_key = config["openai"]["api_key"].as<std::string>();
    std::string model = config["openai"]["model"].as<std::string>();
    //openai::OpenAI client(api_key);

    // PostgreSQL connection string - adjust for your DB
    std::string conn_str = "host=100.73.65.14 port=6432 dbname=pensieve user=postgres password=postgres";
    std::string python_embedder_path = "../python/embedder_repl.py";

    TextProcessor processor(conn_str, "models/sentencepiece.bpe.model", api_key, python_embedder_path);

    //std::cout << "API Key: " << api_key << std::endl;
    std::cout << "Model: " << model << std::endl;
    
    std::cout << "connected to database" << std::endl;

    for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) {
            continue;  // Skip non-regular files
        }

        const std::string path_str = entry.path().string();

        // Handle loading files based on type, defaulting to text file processing
        if (path_str.find("10ks") != std::string::npos) {
            // Check if "10ks" appears anywhere in the full path string
            processor.process_10ks_file(entry.path());
        } else {
            // Default: process as text file (handles .txt, .md, and other text formats)
            processor.process_text_file(entry.path());
        }
    }

    std::cout << "Processing complete." << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: indexer <directory_path>" << std::endl;
        return 1;
    }
    std::string dir_path = argv[1];
    return run_indexer(dir_path);
}
