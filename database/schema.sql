CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    config_json  TEXT NOT NULL,        -- hyperparameters only — NOT ground truth
    test_acc     REAL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS epochs (
    run_id        TEXT NOT NULL,
    epoch         INTEGER NOT NULL,
    train_loss    REAL,
    val_loss      REAL,
    train_acc     REAL,
    val_acc       REAL,
    grad_norm     REAL,
    learning_rate REAL,
    PRIMARY KEY (run_id, epoch),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_epochs_run_id ON epochs(run_id);