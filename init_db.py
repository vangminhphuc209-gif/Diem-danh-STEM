"""
=================================================
  INIT_DB.PY — KHOI TAO DATABASE SQLITE  v2.0
=================================================
BUG FIX:
  - csv_path resolve sai khi chay tu thu muc khac -> dung BASE_DIR
  - conn khong dong khi co exception -> dung try/finally
  - Thieu bang offline_log/offline_violation can cho main.py v2.0

MOI:
  - Them bang: TongKet, offline_log, offline_violation
  - Tu dong migrate schema tu phien ban cu (them cot neu thieu)
  - Hien 5 ban ghi diem danh gan nhat sau khoi tao
  - --reset: xoa sach + tao lai (yeu cau xac nhan)
  - --summary: chi hien thong ke

Cach dung:
    python init_db.py
    python init_db.py --reset
    python init_db.py --summary
"""

import sqlite3
import os
import csv
import argparse

from config import DB_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "students.csv")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# =============================================================================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS HocSinh (
    ma_hs   TEXT PRIMARY KEY,
    ho_ten  TEXT NOT NULL,
    lop     TEXT,
    tao_luc TEXT DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS DiemDanh (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thoi_gian   TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
    ma_hs       TEXT    NOT NULL,
    ho_ten      TEXT,
    trang_thai  TEXT    CHECK(trang_thai IN ('Vao','Ra')),
    da_sync     INTEGER DEFAULT 1,
    FOREIGN KEY (ma_hs) REFERENCES HocSinh(ma_hs)
);
CREATE TABLE IF NOT EXISTS offline_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    thoi_gian  TEXT,
    ma_hs      TEXT,
    ho_ten     TEXT,
    trang_thai TEXT,
    synced     INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS offline_violation (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    thoi_gian  TEXT,
    ma_hs      TEXT,
    ho_ten     TEXT,
    mo_ta      TEXT,
    ten_file   TEXT,
    duong_dan  TEXT,
    synced     INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS TongKet (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ngay         TEXT UNIQUE,
    tong_hs      INTEGER DEFAULT 0,
    co_mat       INTEGER DEFAULT 0,
    vang_mat     INTEGER DEFAULT 0,
    vi_pham      INTEGER DEFAULT 0,
    gio_bat_dau  TEXT,
    gio_ket_thuc TEXT,
    ghi_chu      TEXT
);
"""

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_dd_ma_hs     ON DiemDanh(ma_hs)",
    "CREATE INDEX IF NOT EXISTS idx_dd_thoi_gian ON DiemDanh(thoi_gian)",
    "CREATE INDEX IF NOT EXISTS idx_off_synced   ON offline_log(synced)",
    "CREATE INDEX IF NOT EXISTS idx_vio_synced   ON offline_violation(synced)",
]


# =============================================================================
def init_database(conn):
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    for sql in INDEX_SQL:
        cur.execute(sql)
    conn.commit()
    print("[OK] Da tao / kiem tra cau truc database.")


def migrate_schema(conn):
    """Them cac cot moi neu DB cu chua co (an toan chay nhieu lan)."""
    cur  = conn.cursor()
    cols = [r[1] for r in cur.execute("PRAGMA table_info(DiemDanh)").fetchall()]
    if "da_sync" not in cols:
        cur.execute("ALTER TABLE DiemDanh ADD COLUMN da_sync INTEGER DEFAULT 1")
        conn.commit()
        print("  [MIGRATE] Da them cot 'da_sync' vao DiemDanh.")


def import_students_from_csv(conn):
    """[BUG FIX] Resolve duong dan CSV qua BASE_DIR, bo qua ban ghi trung."""
    if not os.path.exists(CSV_PATH):
        print(f"  [!] Khong tim thay '{CSV_PATH}' — bo qua import.")
        return

    cur = conn.cursor()
    new_count = skipped = 0

    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ma    = row.get("MaHS", "").strip()
            ten   = row.get("HoTen", "").strip()
            lop   = row.get("Lop", "").strip()
            if not ma or not ten:
                continue
            cur.execute(
                "INSERT OR IGNORE INTO HocSinh (ma_hs, ho_ten, lop) VALUES (?,?,?)",
                (ma, ten, lop)
            )
            if cur.rowcount:
                new_count += 1
            else:
                skipped += 1

    conn.commit()
    print(f"  [OK] Import: {new_count} moi, {skipped} da ton tai (bo qua).")


def show_summary(conn):
    cur    = conn.cursor()
    n_hs   = cur.execute("SELECT COUNT(*) FROM HocSinh").fetchone()[0]
    n_log  = cur.execute("SELECT COUNT(*) FROM DiemDanh").fetchone()[0]
    n_off  = cur.execute("SELECT COUNT(*) FROM offline_log  WHERE synced=0").fetchone()[0]
    n_vio  = cur.execute("SELECT COUNT(*) FROM offline_violation WHERE synced=0").fetchone()[0]
    n_sum  = cur.execute("SELECT COUNT(*) FROM TongKet").fetchone()[0]

    print(f"\n  Tong hoc sinh      : {n_hs}")
    print(f"  Log diem danh      : {n_log}")
    print(f"  Offline chua sync  : {n_off} log | {n_vio} vi pham")
    print(f"  Bao cao ngay       : {n_sum} ngay")
    print(f"  Database           : {DB_PATH}")

    recent = cur.execute(
        "SELECT thoi_gian, ma_hs, ho_ten, trang_thai FROM DiemDanh ORDER BY id DESC LIMIT 5"
    ).fetchall()
    if recent:
        print("\n  5 ban ghi gan nhat:")
        for r in recent:
            print(f"    {r[0]}  {r[1]}  {r[2]}  [{r[3]}]")


def reset_database():
    confirm = input(
        "\n[!!] CANH BAO: Xoa TOAN BO du lieu!\n"
        "     Nhap 'XAC NHAN' de tiep tuc: "
    ).strip()
    if confirm != "XAC NHAN":
        print("[OK] Huy bo.")
        return False
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"[OK] Da xoa: {DB_PATH}")
    return True


# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",   action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    print("=" * 54)
    print("  KHOI TAO DATABASE — DIEM DANH QR  v2.0")
    print("=" * 54)

    if args.reset:
        if not reset_database():
            return

    # [BUG FIX] try/finally dam bao conn luon duoc dong
    conn = sqlite3.connect(DB_PATH)
    try:
        init_database(conn)
        migrate_schema(conn)

        if not args.summary:
            print("\n[->] Import hoc sinh:")
            import_students_from_csv(conn)

        show_summary(conn)
    finally:
        conn.close()

    print(f"\n{'=' * 54}")
    print("  [OK] Hoan thanh!")
    print(f"{'=' * 54}")


if __name__ == "__main__":
    main()
