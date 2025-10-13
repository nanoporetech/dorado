import psycopg2
from psycopg2 import sql
from datetime import datetime
from tetra.utilities import get_env_or_raise


# Wrapper around all DB interactions.
class DBWrapper:
    def __init__(self, project: str):
        DB_HOST = get_env_or_raise("BENCHMARK_DB_HOST")
        DB_PASSWRD = get_env_or_raise("BENCHMARK_DB_PASSWORD")
        self._conn = psycopg2.connect(
            host=DB_HOST,
            dbname="benchmarking",
            user="benchmarking_user",
            password=DB_PASSWRD,
        )
        self.project = project

    def create_speed_sql_table(self):
        with self._conn as transaction:
            with transaction.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE {} (
                            name VARCHAR NOT NULL,
                            config VARCHAR NOT NULL,
                            preset VARCHAR NOT NULL,
                            platform VARCHAR NOT NULL,
                            runner VARCHAR NOT NULL,
                            job_id VARCHAR NOT NULL,
                            commit CHAR(40) NOT NULL,
                            date TIMESTAMP NOT NULL,
                            speed BIGINT NOT NULL
                        )
                        """
                    ).format(sql.Identifier(f"{self.project}_speed_measurements"))
                )

    def create_latency_sql_table(self):
        with self._conn as transaction:
            with transaction.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE {} (
                            name VARCHAR NOT NULL,
                            config VARCHAR NOT NULL,
                            preset VARCHAR NOT NULL,
                            platform VARCHAR NOT NULL,
                            runner VARCHAR NOT NULL,
                            job_id VARCHAR NOT NULL,
                            commit CHAR(40) NOT NULL,
                            date TIMESTAMP NOT NULL,
                            as_latency_quartile1 DOUBLE PRECISION NOT NULL,
                            as_latency_median DOUBLE PRECISION NOT NULL,
                            as_latency_quartile3 DOUBLE PRECISION NOT NULL,
                            live_latency_speed DOUBLE PRECISION NOT NULL
                        )
                        """
                    ).format(sql.Identifier(f"{self.project}_latency_measurements"))
                )

    def add_speed_entry(
        self,
        name: str,
        config: str,
        preset: str,
        platform: str,
        runner: str,
        job_id: str,
        commit: str,
        date: datetime,
        speed: int,
    ):
        with self._conn as transaction:
            with transaction.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}
                        VALUES(
                            %(name)s,
                            %(config)s,
                            %(preset)s,
                            %(platform)s,
                            %(runner)s,
                            %(job_id)s,
                            %(commit)s,
                            %(date)s,
                            %(speed)s
                        )
                        """
                    ).format(sql.Identifier(f"{self.project}_speed_measurements")),
                    {
                        "name": name,
                        "config": config,
                        "preset": preset,
                        "platform": platform,
                        "runner": runner,
                        "job_id": job_id,
                        "commit": commit,
                        "date": date,
                        "speed": speed,
                    },
                )

    def add_latency_entry(
        self,
        name: str,
        config: str,
        preset: str,
        platform: str,
        runner: str,
        job_id: str,
        commit: str,
        date: datetime,
        as_latency_quartile1: float,
        as_latency_median: float,
        as_latency_quartile3: float,
        live_latency_speed: float,
    ):
        with self._conn as transaction:
            with transaction.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}
                        VALUES(
                            %(name)s,
                            %(config)s,
                            %(preset)s,
                            %(platform)s,
                            %(runner)s,
                            %(job_id)s,
                            %(commit)s,
                            %(date)s,
                            %(as_latency_quartile1)s,
                            %(as_latency_median)s,
                            %(as_latency_quartile3)s,
                            %(live_latency_speed)s
                        )
                        """
                    ).format(sql.Identifier(f"{self.project}_latency_measurements")),
                    {
                        "name": name,
                        "config": config,
                        "preset": preset,
                        "platform": platform,
                        "runner": runner,
                        "job_id": job_id,
                        "commit": commit,
                        "date": date,
                        "as_latency_quartile1": as_latency_quartile1,
                        "as_latency_median": as_latency_median,
                        "as_latency_quartile3": as_latency_quartile3,
                        "live_latency_speed": live_latency_speed,
                    },
                )

    def get_speed_entries(self, oldest: datetime):
        with self._conn as transaction:
            with transaction.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        SELECT * FROM {}
                        WHERE date >= %(oldest)s
                        """
                    ).format(sql.Identifier(f"{self.project}_speed_measurements")),
                    {"oldest": oldest},
                )
                return [
                    {
                        "name": entry[0],
                        "config": entry[1],
                        "preset": entry[2],
                        "platform": entry[3],
                        "runner": entry[4],
                        "job_id": entry[5],
                        "commit": entry[6],
                        "date": entry[7],
                        "speed": entry[8],
                    }
                    for entry in cursor
                ]

    def get_latency_entries(self, oldest: datetime):
        with self._conn as transaction:
            with transaction.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        SELECT * FROM {}
                        WHERE date >= %(oldest)s
                        """
                    ).format(sql.Identifier(f"{self.project}_latency_measurements")),
                    {"oldest": oldest},
                )
                return [
                    {
                        "name": entry[0],
                        "config": entry[1],
                        "preset": entry[2],
                        "platform": entry[3],
                        "runner": entry[4],
                        "job_id": entry[5],
                        "commit": entry[6],
                        "date": entry[7],
                        "as_latency_quartile1": entry[8],
                        "as_latency_median": entry[9],
                        "as_latency_quartile3": entry[10],
                        "live_latency_speed": entry[11],
                    }
                    for entry in cursor
                ]

    def has_job_entry(self, job_id: str):
        with self._conn as transaction:
            with transaction.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        SELECT EXISTS(
                            (
                                SELECT TRUE FROM {}
                                WHERE job_id = %(job_id)s
                                LIMIT 1
                            ) UNION (
                                SELECT TRUE FROM {}
                                WHERE job_id = %(job_id)s
                                LIMIT 1
                            )
                        )
                        """
                    ).format(
                        sql.Identifier(
                            f"{self.project}_speed_measurements",
                            f"{self.project}_latency_measurements",
                        )
                    ),
                    {"job_id": job_id},
                )
                return cursor.fetchone()[0]
