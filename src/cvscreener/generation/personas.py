"""Deterministic persona matrix for the synthetic candidate pool.

Diversity is engineered here rather than delegated to the LLM. Asked for "50
varied CVs" a model reliably collapses toward the same few archetypes - the same
names, the same three cities, everyone a senior engineer. So the axes that must
vary (language, role, seniority, age, city, university, gender presentation,
layout template) are pinned up front, and the model is left to do the thing it
is genuinely good at: writing believable prose inside those constraints.

A fixed seed makes the whole dataset reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

SEED = 20260814
TOTAL = 50

Language = Literal["es", "en"]


@dataclass(frozen=True)
class Persona:
    cv_id: str
    language: Language
    full_name: str
    gender: str
    role: str
    seniority: str
    years_experience: int
    birth_year: int
    city: str
    country: str
    university: str
    degree_field: str
    template: int
    appearance: str
    signature_skills: tuple[str, ...] = field(default=())

    @property
    def age(self) -> int:
        return 2026 - self.birth_year

    @property
    def role_display(self) -> str:
        """Resolve 'Ingeniero/a Backend' to a real gendered title.

        Feeding the slashed form straight to the model made it echo the slash
        verbatim onto the CV, which no human would ever write.
        """
        if "/a" not in self.role:
            return self.role
        head, _, tail = self.role.partition("/a")
        if self.gender == "female":
            head = head[:-1] + "a" if head.endswith("o") else head + "a"
        return (head + tail).strip()


# --- Vocabulary -----------------------------------------------------------
# Roles mirror the departments Leadtech actually lists on its careers site.
ROLES_ES: list[tuple[str, tuple[str, ...]]] = [
    ("Ingeniero/a Backend", ("python", "django", "postgresql", "docker")),
    ("Ingeniero/a de Datos", ("python", "spark", "airflow", "sql")),
    ("Desarrollador/a Frontend", ("javascript", "react", "typescript", "css")),
    ("Ingeniero/a de Machine Learning", ("python", "pytorch", "mlflow", "scikit-learn")),
    ("Ingeniero/a DevOps", ("kubernetes", "terraform", "aws", "ci/cd")),
    ("Especialista SEO", ("seo", "google analytics", "screaming frog", "sql")),
    ("Especialista SEM/Paid Media", ("google ads", "meta ads", "sql", "looker studio")),
    ("Diseñador/a UX/UI", ("figma", "design systems", "user research", "prototyping")),
    ("Ingeniero/a QA", ("selenium", "pytest", "cypress", "ci/cd")),
    ("Product Manager", ("product discovery", "jira", "sql", "a/b testing")),
    ("Desarrollador/a Mobile", ("kotlin", "swift", "flutter", "rest apis")),
    ("Analista de Datos", ("sql", "python", "tableau", "dbt")),
]

ROLES_EN: list[tuple[str, tuple[str, ...]]] = [
    ("Backend Engineer", ("java", "spring boot", "kafka", "postgresql")),
    ("Data Engineer", ("python", "dbt", "snowflake", "airflow")),
    ("Frontend Engineer", ("typescript", "vue", "javascript", "webpack")),
    ("Machine Learning Engineer", ("python", "tensorflow", "nlp", "kubernetes")),
    ("Site Reliability Engineer", ("go", "prometheus", "kubernetes", "aws")),
    ("SEO Specialist", ("seo", "technical seo", "python", "google search console")),
    ("Growth / Paid Media Manager", ("google ads", "tiktok ads", "sql", "amplitude")),
    ("Product Designer", ("figma", "design systems", "accessibility", "user research")),
    ("QA Automation Engineer", ("playwright", "typescript", "ci/cd", "api testing")),
    ("Product Manager", ("roadmapping", "sql", "a/b testing", "stakeholder management")),
    ("Full-Stack Engineer", ("node.js", "react", "postgresql", "aws")),
    ("Business Intelligence Analyst", ("sql", "power bi", "python", "dbt")),
]

NAMES_ES_F = [
    "Marta Sanchis Ferrer", "Núria Badia Roldán", "Elena Vidal Company",
    "Aitana Serrano Puig", "Clara Bonet Iglesias", "Lucía Cañadas Otero",
    "Irene Mestre Cabrera",
    # New entries (11 additional Spanish names)
    "Valentina Ruiz Sánchez", "Carla Molina García", "Marisa Fernández Díaz",
    "Verónica Suárez López", "Mónica Herrera González", "Rosa García López",
]
NAMES_ES_M = [
    "Guillem Roca Prats", "Álvaro Peñalver Gil", "Sergi Fontcuberta Mas",
    "Rubén Ibarrola Nieto", "Joan Trías Bermúdez", "Óscar Delclós Rivas",
    "Adrián Quesada Lorenzo",
    # New entries (11 additional Spanish names)
    "Borja García Martín", "Iñigo López Ruiz", "Xavi Muñoz López",
    "Felipe Sánchez García", "Raúl Díaz Moreno",
]
NAMES_EN_F = [
    "Priya Raghunathan", "Sinéad O'Halloran", "Amara Nwachukwu",
    "Katarzyna Wilczyńska", "Beatriz Fontoura Lima", "Yuki Matsubara",
    "Hannah Vogelsang",
    # New entries (11 additional English names)
    "Emily Richardson", "Sasha Volkov", "Aisha Patel",
    "Léa Dubois", "Chiara Rossi", "Megan O'Connor",
]
NAMES_EN_M = [
    "Declan Fitzgerald", "Tomás Bekele", "Mateusz Kowalczyk",
    "Rasmus Lindqvist", "Omar Benali", "Nikhil Chandrasekar",
    "Callum Sharpe",
    # New entries (11 additional English names)
    "Marcus Johnson", "Dmitri Sokolov", "Arjun Sharma",
    "Jean-Pierre Leclerc", "Alessandro Ferrari",
]

CITIES_ES = [
    ("Barcelona", "España"), ("Badalona", "España"), ("Madrid", "España"),
    ("Valencia", "España"), ("Girona", "España"), ("Sevilla", "España"),
    ("Bilbao", "España"), ("Zaragoza", "España"), ("Málaga", "España"),
    ("Sabadell", "España"), ("Tarragona", "España"), ("Palma", "España"),
    ("A Coruña", "España"), ("Granada", "España"),
]
CITIES_EN = [
    ("Barcelona", "Spain"), ("London", "United Kingdom"), ("Dublin", "Ireland"),
    ("Berlin", "Germany"), ("Amsterdam", "Netherlands"), ("Lisbon", "Portugal"),
    ("Manchester", "United Kingdom"), ("Kraków", "Poland"), ("Stockholm", "Sweden"),
    ("Toronto", "Canada"), ("Milan", "Italy"), ("Barcelona", "Spain"),
    ("Copenhagen", "Denmark"), ("Vienna", "Austria"),
]

# UPC appears several times on purpose: the brief's own sample question is
# "Which candidate graduated from UPC?", so the dataset must be able to answer it.
UNIS_ES = [
    "Universitat Politècnica de Catalunya (UPC)",
    "Universitat Politècnica de Catalunya (UPC)",
    "Universitat de Barcelona (UB)",
    "Universitat Autònoma de Barcelona (UAB)",
    "Universitat Pompeu Fabra (UPF)",
    "Universidad Politécnica de Madrid (UPM)",
    "Universitat Politècnica de València (UPV)",
    "Universidad de Sevilla",
    "Universidad de Deusto",
    "Universidad de Zaragoza",
    "Universitat Oberta de Catalunya (UOC)",
    "Universitat Ramon Llull - La Salle",
    "Universidade da Coruña",
    "Universidad de Granada",
]
UNIS_EN = [
    "Universitat Politècnica de Catalunya (UPC)",
    "Imperial College London",
    "Trinity College Dublin",
    "Technische Universität Berlin",
    "University of Amsterdam",
    "Instituto Superior Técnico, Lisbon",
    "University of Manchester",
    "AGH University of Science and Technology",
    "KTH Royal Institute of Technology",
    "University of Toronto",
    "Politecnico di Milano",
    "Universitat Pompeu Fabra (UPF)",
    "Danmarks Tekniske Universitet (DTU)",
    "TU Wien",
]

DEGREE_FIELDS = [
    "Computer Engineering", "Software Engineering", "Data Science",
    "Telecommunications Engineering", "Mathematics", "Industrial Engineering",
    "Marketing", "Interaction Design", "Physics", "Business Analytics",
]

# Spanish CVs must name the degree in Spanish; passing the English label through
# produced hybrids like "Ingeniero/a en Computer Engineering".
DEGREE_FIELDS_ES = {
    "Computer Engineering": "Ingeniería Informática",
    "Software Engineering": "Ingeniería del Software",
    "Data Science": "Ciencia de Datos",
    "Telecommunications Engineering": "Ingeniería de Telecomunicaciones",
    "Mathematics": "Matemáticas",
    "Industrial Engineering": "Ingeniería Industrial",
    "Marketing": "Marketing y Comunicación Digital",
    "Interaction Design": "Diseño de Interacción",
    "Physics": "Física",
    "Business Analytics": "Administración y Dirección de Empresas",
}

# Photo-prompt fragments. Varied deliberately so 50 headshots do not read as
# the same person; the diffusion model otherwise regresses hard to a mean face.
APPEARANCE_POOL = [
    "South Asian", "White European", "Black African", "East Asian",
    "Latin American", "Middle Eastern", "Mediterranean", "Northern European",
    "Southeast Asian", "Mixed heritage",
]

SENIORITY_BY_YEARS = [
    (0, 2, "Junior"),
    (3, 5, "Mid-level"),
    (6, 10, "Senior"),
    (11, 30, "Lead"),
]


def _seniority(years: int) -> str:
    for lo, hi, label in SENIORITY_BY_YEARS:
        if lo <= years <= hi:
            return label
    return "Senior"


def build_personas(total: int = TOTAL) -> list[Persona]:
    """Return a reproducible, well-spread set of personas (half ES, half EN).

    The corpus grew from 28 CVs to 50 after the first batch had already been
    generated, and that history is visible in the shape of this function.

    Generating one CV costs ~65 s of local GPU time, so the 28 existing PDFs,
    photos and profile JSONs are worth keeping - regenerating them would be
    half an hour of GPU for an identical result. The pipeline caches per
    ``cv_id``, so a cached artefact is only reused if persona ``cv_01``..
    ``cv_28`` still describe the same 28 people as before.

    A single ``random.Random(SEED)`` would break that. Every shuffle and every
    ``randint`` advances one shared stream, so appending 22 names to the pools
    would reshuffle the first 28 as well: ``cv_07`` would silently become a
    different person with a stale cached PDF attached. Hence two independent
    streams. ``rng`` sees exactly the sequence of calls it saw when there were
    28 personas, and ``rng_new`` - seeded ``SEED + 1`` - handles everything
    added afterwards. The first 28 personas therefore come out byte-identical.
    """
    # Stream 1: drives the original 28. Its call sequence must not change.
    rng = random.Random(SEED)
    personas: list[Persona] = []

    # The original name pools: 7 women + 7 men per language.
    es_names_orig = [
        (n, "female") for n in NAMES_ES_F[:7]
    ] + [(n, "male") for n in NAMES_ES_M[:7]]

    en_names_orig = [
        (n, "female") for n in NAMES_EN_F[:7]
    ] + [(n, "male") for n in NAMES_EN_M[:7]]

    # Shuffled so the ES/EN alternation does not also alternate gender.
    rng.shuffle(es_names_orig)
    rng.shuffle(en_names_orig)

    # Experience is drawn from a fixed pool rather than a range, which
    # guarantees the spread instead of hoping for it: 28 uniform draws would
    # leave holes and clusters.
    years_pool_orig = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 2, 4] * 2
    rng.shuffle(years_pool_orig)

    # Stream 2: everything added when the corpus grew to 50.
    rng_new = random.Random(SEED + 1)

    es_names_new = [
        (n, "female") for n in NAMES_ES_F[7:]
    ] + [(n, "male") for n in NAMES_ES_M[7:]]

    en_names_new = [
        (n, "female") for n in NAMES_EN_F[7:]
    ] + [(n, "male") for n in NAMES_EN_M[7:]]

    rng_new.shuffle(es_names_new)
    rng_new.shuffle(en_names_new)

    years_pool_new = [1, 3, 5, 6, 7, 8, 9, 11, 13, 2, 4, 10, 12, 3, 5, 7, 9, 11, 2, 4, 6, 8]
    rng_new.shuffle(years_pool_new)

    # Appended, never interleaved: the original entries keep their positions,
    # so index i < 28 still resolves to the same name, role, city and university.
    es_names = es_names_orig + es_names_new
    en_names = en_names_orig + en_names_new
    years_pool = years_pool_orig + years_pool_new

    # 14 ES + 14 EN as before, then 11 ES + 11 EN appended - which is what keeps
    # the corpus at an even 25/25 split overall.
    plan: list[Language] = ["es"] * 14 + ["en"] * 14 + ["es"] * 11 + ["en"] * 11

    es_i = en_i = 0
    for idx, lang in enumerate(plan):
        # The birth-year jitter below consumes randomness, so it has to come
        # from the same stream that produced this persona's name.
        rng_active = rng if idx < 28 else rng_new

        if lang == "es":
            name, gender = es_names[es_i]
            role, skills = ROLES_ES[es_i % len(ROLES_ES)]
            city, country = CITIES_ES[es_i % len(CITIES_ES)]
            university = UNIS_ES[es_i % len(UNIS_ES)]
            es_i += 1
        else:
            name, gender = en_names[en_i]
            role, skills = ROLES_EN[en_i % len(ROLES_EN)]
            city, country = CITIES_EN[en_i % len(CITIES_EN)]
            university = UNIS_EN[en_i % len(UNIS_EN)]
            en_i += 1

        years = years_pool[idx]
        # Graduate around 22-24, so age tracks experience with a little jitter.
        birth_year = 2026 - (22 + years + rng_active.randint(0, 6))

        personas.append(
            Persona(
                cv_id=f"cv_{idx + 1:02d}",
                language=lang,
                full_name=name,
                gender=gender,
                role=role,
                seniority=_seniority(years),
                years_experience=years,
                birth_year=birth_year,
                city=city,
                country=country,
                university=university,
                degree_field=DEGREE_FIELDS[idx % len(DEGREE_FIELDS)],
                template=idx % 3,
                appearance=APPEARANCE_POOL[idx % len(APPEARANCE_POOL)],
                signature_skills=skills,
            )
        )
    return personas


if __name__ == "__main__":  # quick inspection aid
    import sys

    # Windows consoles default to cp1252, which cannot encode names like
    # "Wilczyńska". Everything downstream is UTF-8, so make stdout match.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    for p in build_personas():
        print(
            f"{p.cv_id} {p.language} t{p.template} {p.age:>2}y "
            f"{p.years_experience:>2}exp {p.full_name:<26} {p.role:<38} {p.city}"
        )
