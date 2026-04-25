echo off

python ingest.py ^
    --category="['core_clinical', 'basic_biology', 'pharmacology', 'psychiatry']" ^
    --num_samples=100 ^
    --start_idx=0 

@REM --category="['core_clinical', 'basic_biology', 'pharmacology', 'psychiatry']" ^