import os
import duckdb # type: ignore
from config import DATA_LAKE_DIR

DATA_ROOT = str(DATA_LAKE_DIR)
ANALYSES_DIR = os.path.join(DATA_ROOT, "analyses", "year=*", "month=*", "stream=*", "*.parquet")
DUCKDB_PATH = os.path.join(DATA_ROOT, "duckdb", "warehouse.duckdb")
os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)

def get_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=DUCKDB_PATH)
    con.execute("INSTALL httpfs; LOAD httpfs;")
    return con

def create_views():
    con = get_con()
    con.execute(f"""
        CREATE OR REPLACE VIEW analyses AS
        SELECT
            key,
            ts_utc::TIMESTAMP AS ts_utc,
            year, month, stream,
            narrator_id, narrator_name,
            source_platform, source_external_id, source_title,
            run_id, pipeline_version, asr_model, scoring_model,
            final_score, classification, total_segments, total_duration,
            audio_total, text_total,
            raw_json
        FROM read_parquet('{ANALYSES_DIR}');
    """)
    con.close()

def query(sql: str):
    con = get_con()
    try:
        return con.execute(sql).df()
    finally:
        con.close()


if __name__ == "__main__":
    create_views()

    print(query("SELECT COUNT(*) AS total FROM analyses;"))