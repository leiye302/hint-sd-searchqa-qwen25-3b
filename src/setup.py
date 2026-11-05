"""Bridge for unchanged shared QA preprocessing/retrieval source checks."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hint_searchqa.common import verify_sources as verify


if __name__ == '__main__':
    from hint_searchqa.setup import main
    main()
