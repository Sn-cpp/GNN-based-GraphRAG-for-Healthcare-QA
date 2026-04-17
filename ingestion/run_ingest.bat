echo off

python ingest.py ^
    --category="['core_clinical', 'basic_biology', 'pharmacology', 'psychiatry']" ^
    --num_samples=200

@REM --category="['core_clinical', 'basic_biology', 'pharmacology', 'psychiatry']" ^