-- Schema for the ML Training Log Diagnostician database.
-- Two tables: one row per run in `runs`, one row per epoch in `epochs`.

CREATE TABLE IF NOT EXISTS runs (
    run_id                TEXT PRIMARY KEY,
    label                 TEXT NOT NULL,        -- 'healthy' or 'problem'
    injected_problem_type TEXT NOT NULL,        -- 'none', 'lr_too_high', 'lr_too_low',
                                                 -- 'overfitting', 'vanishing_gradients',
                                                 -- 'label_noise', 'frozen_layer'
    config_json           TEXT NOT NULL,        -- full hyperparameter config, as JSON text
    test_acc              REAL,
    created_at            TEXT NOT NULL
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