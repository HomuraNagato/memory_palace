package main

import (
    "database/sql"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "os/exec"

    _ "github.com/lib/pq"
)

type requestPayload struct {
    Dir string `json:"dir"`
}

type responsePayload struct {
    Count int `json:"count"`
}

func main() {
    // Set up DB connection pool
    connStr := "host=localhost port=5432 user=your_user password=your_password dbname=your_db sslmode=disable"
    db, err := sql.Open("postgres", connStr)
    if err != nil {
        log.Fatalf("DB connection failed: %v", err)
    }
    defer db.Close()

    http.HandleFunc("/index", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost {
            http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
            return
        }

        var payload requestPayload
        if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
            http.Error(w, "Invalid JSON", http.StatusBadRequest)
            return
        }
        if payload.Dir == "" {
            http.Error(w, "Missing 'dir' parameter", http.StatusBadRequest)
            return
        }

        // Call C++ worker as external process
        cmd := exec.Command("./worker", payload.Dir)
        output, err := cmd.CombinedOutput()
        if err != nil {
            msg := fmt.Sprintf("Worker failed: %v\nOutput: %s", err, string(output))
            http.Error(w, msg, http.StatusInternalServerError)
            return
        }

        // Query count of documents
        var count int
        err = db.QueryRow("SELECT COUNT(*) FROM documents;").Scan(&count)
        if err != nil {
            http.Error(w, "Failed to fetch count", http.StatusInternalServerError)
            return
        }

        resp := responsePayload{Count: count}
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(resp)
    })

    fmt.Println("Server starting at :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
