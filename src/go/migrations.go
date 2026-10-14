package main

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"embed"
	"encoding/hex"
	"fmt"
	"io"
	"io/fs"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
)

const migrationAdvisoryLock int64 = 0x4d50414c414345

var migrationNamePattern = regexp.MustCompile(`^(\d{6})_([a-z0-9_]+)\.sql$`)

//go:embed migrations/*.sql
var embeddedMigrations embed.FS

type migration struct {
	Version  int64
	Name     string
	SQL      string
	Checksum string
}

type appliedMigration struct {
	Name     string
	Checksum string
}

func loadMigrations(files fs.FS) ([]migration, error) {
	entries, err := fs.ReadDir(files, "migrations")
	if err != nil {
		return nil, fmt.Errorf("read migrations: %w", err)
	}
	migrations := make([]migration, 0, len(entries))
	seen := make(map[int64]string, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		matches := migrationNamePattern.FindStringSubmatch(entry.Name())
		if matches == nil {
			return nil, fmt.Errorf("invalid migration filename %q", entry.Name())
		}
		version, err := strconv.ParseInt(matches[1], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parse migration version %q: %w", entry.Name(), err)
		}
		if previous, exists := seen[version]; exists {
			return nil, fmt.Errorf("migration version %06d is duplicated by %q and %q", version, previous, entry.Name())
		}
		body, err := fs.ReadFile(files, filepath.ToSlash(filepath.Join("migrations", entry.Name())))
		if err != nil {
			return nil, fmt.Errorf("read migration %q: %w", entry.Name(), err)
		}
		digest := sha256.Sum256(body)
		migrations = append(migrations, migration{
			Version:  version,
			Name:     matches[2],
			SQL:      string(body),
			Checksum: hex.EncodeToString(digest[:]),
		})
		seen[version] = entry.Name()
	}
	sort.Slice(migrations, func(i, j int) bool { return migrations[i].Version < migrations[j].Version })
	if len(migrations) == 0 {
		return nil, fmt.Errorf("no migrations found")
	}
	return migrations, nil
}

func migrate(db *sql.DB) error {
	migrations, err := loadMigrations(embeddedMigrations)
	if err != nil {
		return err
	}
	ctx := context.Background()
	conn, err := db.Conn(ctx)
	if err != nil {
		return fmt.Errorf("open migration connection: %w", err)
	}
	defer conn.Close()

	if _, err := conn.ExecContext(ctx, `SELECT pg_advisory_lock($1)`, migrationAdvisoryLock); err != nil {
		return fmt.Errorf("lock migrations: %w", err)
	}
	defer conn.ExecContext(context.Background(), `SELECT pg_advisory_unlock($1)`, migrationAdvisoryLock)

	if _, err := conn.ExecContext(ctx, `
		CREATE TABLE IF NOT EXISTS memory_palace_schema_migrations (
			version BIGINT PRIMARY KEY,
			name TEXT NOT NULL,
			checksum TEXT NOT NULL,
			applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`); err != nil {
		return fmt.Errorf("create migration history: %w", err)
	}
	applied, err := readAppliedMigrations(ctx, conn)
	if err != nil {
		return err
	}

	available := make(map[int64]migration, len(migrations))
	for _, item := range migrations {
		available[item.Version] = item
	}
	for version, recorded := range applied {
		if _, ok := available[version]; !ok {
			return fmt.Errorf("applied migration %06d_%s is missing from the binary", version, recorded.Name)
		}
	}
	for _, item := range migrations {
		if recorded, ok := applied[item.Version]; ok {
			if recorded.Name != item.Name || recorded.Checksum != item.Checksum {
				return fmt.Errorf("migration %06d_%s.sql differs from the applied migration", item.Version, item.Name)
			}
			continue
		}
		tx, err := conn.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("begin migration %06d_%s: %w", item.Version, item.Name, err)
		}
		if _, err := tx.ExecContext(ctx, item.SQL); err != nil {
			tx.Rollback()
			return fmt.Errorf("apply migration %06d_%s: %w", item.Version, item.Name, err)
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO memory_palace_schema_migrations (version, name, checksum)
			VALUES ($1, $2, $3)
		`, item.Version, item.Name, item.Checksum); err != nil {
			tx.Rollback()
			return fmt.Errorf("record migration %06d_%s: %w", item.Version, item.Name, err)
		}
		if err := tx.Commit(); err != nil {
			return fmt.Errorf("commit migration %06d_%s: %w", item.Version, item.Name, err)
		}
	}
	return nil
}

func readAppliedMigrations(ctx context.Context, queryer interface {
	QueryContext(context.Context, string, ...any) (*sql.Rows, error)
}) (map[int64]appliedMigration, error) {
	rows, err := queryer.QueryContext(ctx, `
		SELECT version, name, checksum
		FROM memory_palace_schema_migrations
		ORDER BY version
	`)
	if err != nil {
		return nil, fmt.Errorf("read migration history: %w", err)
	}
	defer rows.Close()
	applied := make(map[int64]appliedMigration)
	for rows.Next() {
		var version int64
		var item appliedMigration
		if err := rows.Scan(&version, &item.Name, &item.Checksum); err != nil {
			return nil, fmt.Errorf("scan migration history: %w", err)
		}
		applied[version] = item
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate migration history: %w", err)
	}
	return applied, nil
}

func writeMigrationStatus(db *sql.DB, output io.Writer) error {
	migrations, err := loadMigrations(embeddedMigrations)
	if err != nil {
		return err
	}
	var historyExists bool
	if err := db.QueryRow(`SELECT to_regclass('memory_palace_schema_migrations') IS NOT NULL`).Scan(&historyExists); err != nil {
		return err
	}
	applied := map[int64]appliedMigration{}
	if historyExists {
		applied, err = readAppliedMigrations(context.Background(), db)
		if err != nil {
			return err
		}
	}
	for _, item := range migrations {
		state := "pending"
		if recorded, ok := applied[item.Version]; ok {
			state = "applied"
			if recorded.Name != item.Name || recorded.Checksum != item.Checksum {
				state = "checksum-mismatch"
			}
		}
		if _, err := fmt.Fprintf(output, "%06d  %-18s  %s\n", item.Version, state, item.Name); err != nil {
			return err
		}
	}
	return nil
}
