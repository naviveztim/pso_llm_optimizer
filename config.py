from pathlib import Path
from typing import List

from utils import PersonaProfile


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE_PATH = REPO_ROOT / "data" / "kaufland_prices_by_category_various_countries.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "family_pso_plan.json"
DEFAULT_COPILOT_MODEL = "GPT-5.6 Luna"
DEFAULT_COPILOT_TOKEN_ENV = "COPILOT_GITHUB_TOKEN"
DEFAULT_NUM_ITERATIONS = 10
DEFAULT_COUNTRY = "DE"
MAX_CANDIDATES = 220
DEFAULT_NUMBER_OF_PARTICLES = 3
DEFAULT_STAGNATION_WINDOW = 5
DEFAULT_MIN_DELTA = 0.05
SEED = 42

# Internal influence strengths. The LLM receives only their normalized shares,
# not these implementation-specific names or PSO terminology.
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


def make_profiles() -> List[PersonaProfile]:
	"""Return the family personas and their basket requirements."""
	return [
		PersonaProfile(
			name="father",
			role="father",
			keywords=["beer", "bier", "meat", "fleisch", "grill", "sausage", "wurst"],
			basket_size=12,
			min_preferred_items=4,
		),
		PersonaProfile(
			name="mother",
			role="mother",
			keywords=["gemuese", "gemuse", "obst", "salat", "bio", "tomaten", "gurke"],
			basket_size=12,
			min_preferred_items=4,
		),
		PersonaProfile(
			name="daughter",
			role="daughter",
			keywords=["chocolate", "schokolade", "candy", "bonbon", "keks", "ice", "dessert"],
			basket_size=10,
			min_preferred_items=3,
		),
		PersonaProfile(
			name="son",
			role="son",
			keywords=["chocolate", "schokolade", "snack", "chips", "candy", "cola"],
			basket_size=10,
			min_preferred_items=3,
		),
		PersonaProfile(
			name="mother_in_law",
			role="mother in law",
			keywords=["detergent", "clean", "reiniger", "spul", "putz", "haushalt", "wasch"],
			basket_size=11,
			min_preferred_items=3,
		),
	]

