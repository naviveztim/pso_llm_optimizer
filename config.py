from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE_PATH = REPO_ROOT / "data" / "kaufland_prices_by_category_various_countries.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "family_pso_plan.json"
DEFAULT_MODEL = "GPT-5.6 Luna"
DEFAULT_TOKEN_ENV = "COPILOT_GITHUB_TOKEN"
DEFAULT_NUM_ITERATIONS = 10
DEFAULT_COUNTRY = "DE"
MAX_CANDIDATES = 220
DEFAULT_NUMBER_OF_PARTICLES = 3
DEFAULT_STAGNATION_WINDOW = 5
DEFAULT_MIN_DELTA = 0.05
SEED = 42
PSO_INERTIA = 0.7
PSO_COGNITIVE = 0.8
PSO_SOCIAL = 1.8
PSO_EXPLORATION = 0.2
HEALTHY_KEYWORDS = (
	"bio",
	"gemuese",
	"gemuse",
	"obst",
	"salat",
	"tomaten",
	"gurke",
	"vollkorn",
	"hafer",
	"natur",
	"fresh",
	"frisch",
)
