package main

import (
	"database/sql"
	"fmt"
	"log"
	"os"

	_ "github.com/lib/pq"
)

func main() {
	// Database connection
	connStr := envOrDefault(
		"MEMORY_PALACE_DATABASE_URL",
		"host=100.73.65.14 port=5438 dbname=pensieve user=postgres password=postgres sslmode=disable",
	)
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// Test connection
	if err := db.Ping(); err != nil {
		log.Fatalf("Failed to ping database: %v", err)
	}
	log.Println("Database connected successfully")

	// Test 1: Simple COUNT query
	var count int
	err = db.QueryRow("SELECT COUNT(*) FROM embeddings").Scan(&count)
	if err != nil {
		log.Fatalf("COUNT query failed: %v", err)
	}
	log.Printf("Total rows in embeddings table: %d", count)

	// Test 2: Fetch multiple rows with simple query
	log.Println("\n=== Test 2: Simple SELECT with LIMIT 15 ===")
	rows, err := db.Query("SELECT file_id, row_num, chunk_idx FROM embeddings LIMIT 15")
	if err != nil {
		log.Fatalf("Simple query failed: %v", err)
	}
	defer rows.Close()

	rowCount := 0
	for rows.Next() {
		rowCount++
		var fileID, rowNum, chunkIdx int
		if err := rows.Scan(&fileID, &rowNum, &chunkIdx); err != nil {
			log.Printf("Error scanning row %d: %v", rowCount, err)
			continue
		}
		log.Printf("Row %d: file_id=%d, row_num=%d, chunk_idx=%d", rowCount, fileID, rowNum, chunkIdx)
	}
	if err := rows.Err(); err != nil {
		log.Printf("Row iteration error: %v", err)
	}
	log.Printf("Simple query returned %d rows\n", rowCount)

	// Test 3: Fetch first embedding to test with
	log.Println("\n=== Test 3: Get sample embedding for vector query ===")
	var sampleEmb string
	err = db.QueryRow("SELECT emb::text FROM embeddings LIMIT 1").Scan(&sampleEmb)
	if err != nil {
		log.Fatalf("Failed to get sample embedding: %v", err)
	}
	log.Printf("Sample embedding length: %d characters", len(sampleEmb))
	log.Printf("Sample embedding (first 100 chars): %s...", sampleEmb[:100])

	// Test 4: Vector similarity query WITH ORDER BY
	log.Println("\n=== Test 4: Vector similarity query WITH ORDER BY ===")
	query := fmt.Sprintf(`
		SELECT file_id, row_num, chunk_idx, 1 - (emb <=> '%s'::vector) as similarity
		FROM embeddings
		ORDER BY emb <=> '%s'::vector
		LIMIT 15
	`, sampleEmb, sampleEmb)

	log.Printf("Query length: %d characters", len(query))
	log.Printf("Query (first 200 chars): %s...", query[:200])

	rows2, err := db.Query(query)
	if err != nil {
		log.Fatalf("Vector query failed: %v", err)
	}
	defer rows2.Close()

	rowCount2 := 0
	for rows2.Next() {
		rowCount2++
		var fileID, rowNum, chunkIdx int
		var similarity float64
		if err := rows2.Scan(&fileID, &rowNum, &chunkIdx, &similarity); err != nil {
			log.Printf("Error scanning row %d: %v", rowCount2, err)
			continue
		}
		log.Printf("Row %d: file_id=%d, row_num=%d, chunk_idx=%d, similarity=%.6f",
			rowCount2, fileID, rowNum, chunkIdx, similarity)
	}
	if err := rows2.Err(); err != nil {
		log.Printf("Row iteration error: %v", err)
	}
	log.Printf("Vector query WITH ORDER BY returned %d rows\n", rowCount2)

	// Test 5: Vector similarity query WITHOUT ORDER BY
	log.Println("\n=== Test 5: Vector similarity query WITHOUT ORDER BY ===")
	query2 := fmt.Sprintf(`
		SELECT file_id, row_num, chunk_idx, 1 - (emb <=> '%s'::vector) as similarity
		FROM embeddings
		LIMIT 15
	`, sampleEmb)

	log.Printf("Query length: %d characters", len(query2))

	rows3, err := db.Query(query2)
	if err != nil {
		log.Fatalf("Vector query without ORDER BY failed: %v", err)
	}
	defer rows3.Close()

	rowCount3 := 0
	for rows3.Next() {
		rowCount3++
		var fileID, rowNum, chunkIdx int
		var similarity float64
		if err := rows3.Scan(&fileID, &rowNum, &chunkIdx, &similarity); err != nil {
			log.Printf("Error scanning row %d: %v", rowCount3, err)
			continue
		}
		log.Printf("Row %d: file_id=%d, row_num=%d, chunk_idx=%d, similarity=%.6f",
			rowCount3, fileID, rowNum, chunkIdx, similarity)
	}
	if err := rows3.Err(); err != nil {
		log.Printf("Row iteration error: %v", err)
	}
	log.Printf("Vector query WITHOUT ORDER BY returned %d rows\n", rowCount3)

	// Test 6: Vector similarity query using CTE (Common Table Expression)
	log.Println("\n=== Test 6: Vector similarity query using CTE ===")
	query3 := fmt.Sprintf(`
		WITH query_vector AS (
			SELECT '%s'::vector as vec
		)
		SELECT file_id, row_num, chunk_idx, 1 - (emb <=> query_vector.vec) as similarity
		FROM embeddings, query_vector
		ORDER BY emb <=> query_vector.vec
		LIMIT 15
	`, sampleEmb)

	log.Printf("Query length: %d characters", len(query3))

	rows4, err := db.Query(query3)
	if err != nil {
		log.Fatalf("Vector query with CTE failed: %v", err)
	}
	defer rows4.Close()

	rowCount4 := 0
	for rows4.Next() {
		rowCount4++
		var fileID, rowNum, chunkIdx int
		var similarity float64
		if err := rows4.Scan(&fileID, &rowNum, &chunkIdx, &similarity); err != nil {
			log.Printf("Error scanning row %d: %v", rowCount4, err)
			continue
		}
		log.Printf("Row %d: file_id=%d, row_num=%d, chunk_idx=%d, similarity=%.6f",
			rowCount4, fileID, rowNum, chunkIdx, similarity)
	}
	if err := rows4.Err(); err != nil {
		log.Printf("Row iteration error: %v", err)
	}
	log.Printf("Vector query with CTE returned %d rows\n", rowCount4)

	// Test 7: Check ivfflat.probes setting
	log.Println("\n=== Test 7: Check and set ivfflat.probes ===")
	var probes string
	err = db.QueryRow("SHOW ivfflat.probes").Scan(&probes)
	if err != nil {
		log.Printf("Failed to get ivfflat.probes: %v", err)
	} else {
		log.Printf("Current ivfflat.probes: %s", probes)
	}

	// Set ivfflat.probes to a higher value (default is 1, max is lists count which is 100)
	_, err = db.Exec("SET ivfflat.probes = 10")
	if err != nil {
		log.Printf("Failed to set ivfflat.probes: %v", err)
	} else {
		log.Println("Set ivfflat.probes to 10")
	}

	// Test 8: Retry vector query with CTE after increasing ivfflat.probes
	log.Println("\n=== Test 8: Vector similarity query with CTE after increasing probes ===")
	query4 := fmt.Sprintf(`
		WITH query_vector AS (
			SELECT '%s'::vector as vec
		)
		SELECT file_id, row_num, chunk_idx, 1 - (emb <=> query_vector.vec) as similarity
		FROM embeddings, query_vector
		ORDER BY emb <=> query_vector.vec
		LIMIT 15
	`, sampleEmb)

	rows5, err := db.Query(query4)
	if err != nil {
		log.Fatalf("Vector query with CTE (after probes) failed: %v", err)
	}
	defer rows5.Close()

	rowCount5 := 0
	for rows5.Next() {
		rowCount5++
		var fileID, rowNum, chunkIdx int
		var similarity float64
		if err := rows5.Scan(&fileID, &rowNum, &chunkIdx, &similarity); err != nil {
			log.Printf("Error scanning row %d: %v", rowCount5, err)
			continue
		}
		log.Printf("Row %d: file_id=%d, row_num=%d, chunk_idx=%d, similarity=%.6f",
			rowCount5, fileID, rowNum, chunkIdx, similarity)
	}
	if err := rows5.Err(); err != nil {
		log.Printf("Row iteration error: %v", err)
	}
	log.Printf("Vector query with CTE (probes=10) returned %d rows\n", rowCount5)
}

func envOrDefault(key string, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}

	return value
}
