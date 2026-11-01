# Compare top 1-10 retrieved passages on an editable 200-claim diagnostic.
#
# Run from the repository root::
#
#     .\.venv\Scripts\python.exe "Python Scripts\evaluate_general_claims.py"
#
# Edit CLAIMS below to repeat this experiment. Related true/false examples must
# share a group and split. The selection split chooses k before the check split
# is evaluated; check results never change that selection. This runner uses only
# local retrieval/BERT assets and does not load an LLM client or read API keys.

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["SUPPORTS", "REFUTES"]
KS = list(range(1, 11))
EXPECTED_CLAIM_COUNT = 200
EXPECTED_CATEGORY_COUNT = 10
DEFAULT_OUTPUT_DIR = ROOT / "Evaluation" / "general_claims_topk"
CLIENT_TIMEZONE = "Africa/Windhoek"
PROBABILITY_TOLERANCE = 1e-7

# BEGIN EDITABLE CLAIMS
# Edit claim text, expected label and source here; each pair stays in one split.
# Keep 200 claims: ten topics, 20 per topic, 100 per label, and 100 per split.
# Each topic/split has five SUPPORTS and five REFUTES claims.
CLAIMS = [
    # Capitals
    {"id": "CAP01S", "category": "Capitals", "group": "CAP01", "split": "selection",
     "claim": "The capital of Canada is Ottawa.",
     "expected": "SUPPORTS", "source": "https://www.canada.ca/en/immigration-refugees-citizenship/corporate/publications-manuals/discover-canada/read-online/canadas-regions.html"},
    {"id": "CAP01R", "category": "Capitals", "group": "CAP01", "split": "selection",
     "claim": "The capital of Canada is Toronto.",
     "expected": "REFUTES", "source": "https://www.canada.ca/en/immigration-refugees-citizenship/corporate/publications-manuals/discover-canada/read-online/canadas-regions.html"},
    {"id": "CAP02S", "category": "Capitals", "group": "CAP02", "split": "selection",
     "claim": "The capital of Kenya is Nairobi.",
     "expected": "SUPPORTS", "source": "https://www.eac.go.ke/about-kenya"},
    {"id": "CAP02R", "category": "Capitals", "group": "CAP02", "split": "selection",
     "claim": "The capital of Kenya is Mombasa.",
     "expected": "REFUTES", "source": "https://www.eac.go.ke/about-kenya"},
    {"id": "CAP03S", "category": "Capitals", "group": "CAP03", "split": "selection",
     "claim": "The capital of Germany is Berlin.",
     "expected": "SUPPORTS", "source": "https://www.berlin.de/rbmskzl/en/"},
    {"id": "CAP03R", "category": "Capitals", "group": "CAP03", "split": "selection",
     "claim": "The capital of Germany is Munich.",
     "expected": "REFUTES", "source": "https://www.berlin.de/rbmskzl/en/"},
    {"id": "CAP04S", "category": "Capitals", "group": "CAP04", "split": "check",
     "claim": "The capital of Spain is Madrid.",
     "expected": "SUPPORTS", "source": "https://www.lamoncloa.gob.es/lang/en/espana/leyfundamental/paginas/titulo_preliminar.aspx"},
    {"id": "CAP04R", "category": "Capitals", "group": "CAP04", "split": "check",
     "claim": "The capital of Spain is Barcelona.",
     "expected": "REFUTES", "source": "https://www.lamoncloa.gob.es/lang/en/espana/leyfundamental/paginas/titulo_preliminar.aspx"},
    {"id": "CAP05S", "category": "Capitals", "group": "CAP05", "split": "check",
     "claim": "The capital of New Zealand is Wellington.",
     "expected": "SUPPORTS", "source": "https://teara.govt.nz/en/capital-city"},
    {"id": "CAP05R", "category": "Capitals", "group": "CAP05", "split": "check",
     "claim": "The capital of New Zealand is Auckland.",
     "expected": "REFUTES", "source": "https://teara.govt.nz/en/capital-city"},
    {"id": "CAP06S", "category": "Capitals", "group": "CAP06", "split": "selection",
     "claim": "The capital of Portugal is Lisbon.",
     "expected": "SUPPORTS", "source": "https://www.visitportugal.com/en/NR/exeres/98BE771F-120C-4A83-A18B-DFC8C7907F5B"},
    {"id": "CAP06R", "category": "Capitals", "group": "CAP06", "split": "selection",
     "claim": "The capital of Portugal is Porto.",
     "expected": "REFUTES", "source": "https://www.visitportugal.com/en/NR/exeres/98BE771F-120C-4A83-A18B-DFC8C7907F5B"},
    {"id": "CAP07S", "category": "Capitals", "group": "CAP07", "split": "selection",
     "claim": "The capital of Argentina is Buenos Aires.",
     "expected": "SUPPORTS", "source": "https://www.argentina.gob.ar/caba"},
    {"id": "CAP07R", "category": "Capitals", "group": "CAP07", "split": "selection",
     "claim": "The capital of Argentina is Cordoba.",
     "expected": "REFUTES", "source": "https://www.argentina.gob.ar/caba"},
    {"id": "CAP08S", "category": "Capitals", "group": "CAP08", "split": "check",
     "claim": "The capital of Turkey is Ankara.",
     "expected": "SUPPORTS", "source": "https://www.iletisim.gov.tr/ENGLISH/haberler/detay/president-erdogan-marks-99th-anniversary-of-ankaras-becoming-the-capital"},
    {"id": "CAP08R", "category": "Capitals", "group": "CAP08", "split": "check",
     "claim": "The capital of Turkey is Istanbul.",
     "expected": "REFUTES", "source": "https://www.iletisim.gov.tr/ENGLISH/haberler/detay/president-erdogan-marks-99th-anniversary-of-ankaras-becoming-the-capital"},
    {"id": "CAP09S", "category": "Capitals", "group": "CAP09", "split": "check",
     "claim": "The capital of Thailand is Bangkok.",
     "expected": "SUPPORTS", "source": "https://thailand.go.th/page/capital-city"},
    {"id": "CAP09R", "category": "Capitals", "group": "CAP09", "split": "check",
     "claim": "The capital of Thailand is Chiang Mai.",
     "expected": "REFUTES", "source": "https://thailand.go.th/page/capital-city"},
    {"id": "CAP10S", "category": "Capitals", "group": "CAP10", "split": "check",
     "claim": "The capital of Namibia is Windhoek.",
     "expected": "SUPPORTS", "source": "https://namibiaembassyusa.org/about-namibia/"},
    {"id": "CAP10R", "category": "Capitals", "group": "CAP10", "split": "check",
     "claim": "The capital of Namibia is Walvis Bay.",
     "expected": "REFUTES", "source": "https://namibiaembassyusa.org/about-namibia/"},
    # Landmarks and locations
    {"id": "LOC01S", "category": "Landmarks and locations", "group": "LOC01", "split": "selection",
     "claim": "The Taj Mahal is located in Agra.",
     "expected": "SUPPORTS", "source": "https://whc.unesco.org/en/list/252/"},
    {"id": "LOC01R", "category": "Landmarks and locations", "group": "LOC01", "split": "selection",
     "claim": "The Taj Mahal is located in Mumbai.",
     "expected": "REFUTES", "source": "https://whc.unesco.org/en/list/252/"},
    {"id": "LOC02S", "category": "Landmarks and locations", "group": "LOC02", "split": "selection",
     "claim": "The Statue of Liberty stands on Liberty Island.",
     "expected": "SUPPORTS", "source": "https://www.nps.gov/stli/planyourvisit/frequently-asked-questions-statue-of-liberty.htm"},
    {"id": "LOC02R", "category": "Landmarks and locations", "group": "LOC02", "split": "selection",
     "claim": "The Statue of Liberty stands on Ellis Island.",
     "expected": "REFUTES", "source": "https://www.nps.gov/stli/planyourvisit/frequently-asked-questions-statue-of-liberty.htm"},
    {"id": "LOC03S", "category": "Landmarks and locations", "group": "LOC03", "split": "selection",
     "claim": "The Colosseum is located in Rome.",
     "expected": "SUPPORTS", "source": "https://colosseo.it/en/area/colosseo/"},
    {"id": "LOC03R", "category": "Landmarks and locations", "group": "LOC03", "split": "selection",
     "claim": "The Colosseum is located in Athens.",
     "expected": "REFUTES", "source": "https://colosseo.it/en/area/colosseo/"},
    {"id": "LOC04S", "category": "Landmarks and locations", "group": "LOC04", "split": "check",
     "claim": "Machu Picchu is located in Peru.",
     "expected": "SUPPORTS", "source": "https://whc.unesco.org/en/list/274/"},
    {"id": "LOC04R", "category": "Landmarks and locations", "group": "LOC04", "split": "check",
     "claim": "Machu Picchu is located in Mexico.",
     "expected": "REFUTES", "source": "https://whc.unesco.org/en/list/274/"},
    {"id": "LOC05S", "category": "Landmarks and locations", "group": "LOC05", "split": "check",
     "claim": "Stonehenge is located in Wiltshire.",
     "expected": "SUPPORTS", "source": "https://whc.unesco.org/en/list/373/"},
    {"id": "LOC05R", "category": "Landmarks and locations", "group": "LOC05", "split": "check",
     "claim": "Stonehenge is located in Yorkshire.",
     "expected": "REFUTES", "source": "https://whc.unesco.org/en/list/373/"},
    {"id": "LOC06S", "category": "Landmarks and locations", "group": "LOC06", "split": "selection",
     "claim": "The Great Wall of China is located in China.",
     "expected": "SUPPORTS", "source": "https://whc.unesco.org/en/list/438/"},
    {"id": "LOC06R", "category": "Landmarks and locations", "group": "LOC06", "split": "selection",
     "claim": "The Great Wall of China is located in India.",
     "expected": "REFUTES", "source": "https://whc.unesco.org/en/list/438/"},
    {"id": "LOC07S", "category": "Landmarks and locations", "group": "LOC07", "split": "selection",
     "claim": "The Parthenon in Greece is located in Athens.",
     "expected": "SUPPORTS", "source": "https://whc.unesco.org/en/list/404/"},
    {"id": "LOC07R", "category": "Landmarks and locations", "group": "LOC07", "split": "selection",
     "claim": "The Parthenon in Greece is located in Sparta.",
     "expected": "REFUTES", "source": "https://whc.unesco.org/en/list/404/"},
    {"id": "LOC08S", "category": "Landmarks and locations", "group": "LOC08", "split": "check",
     "claim": "The Sydney Opera House is located in Sydney.",
     "expected": "SUPPORTS", "source": "https://www.nsw.gov.au/visiting-and-exploring-nsw/locations-and-attractions/sydney-opera-house"},
    {"id": "LOC08R", "category": "Landmarks and locations", "group": "LOC08", "split": "check",
     "claim": "The Sydney Opera House is located in Melbourne.",
     "expected": "REFUTES", "source": "https://www.nsw.gov.au/visiting-and-exploring-nsw/locations-and-attractions/sydney-opera-house"},
    {"id": "LOC09S", "category": "Landmarks and locations", "group": "LOC09", "split": "check",
     "claim": "The ancient city of Petra is located in Jordan.",
     "expected": "SUPPORTS", "source": "https://whc.unesco.org/en/list/326/"},
    {"id": "LOC09R", "category": "Landmarks and locations", "group": "LOC09", "split": "check",
     "claim": "The ancient city of Petra is located in Lebanon.",
     "expected": "REFUTES", "source": "https://whc.unesco.org/en/list/326/"},
    {"id": "LOC10S", "category": "Landmarks and locations", "group": "LOC10", "split": "check",
     "claim": "Mount Fuji is located in Japan.",
     "expected": "SUPPORTS", "source": "https://whc.unesco.org/en/list/1418/"},
    {"id": "LOC10R", "category": "Landmarks and locations", "group": "LOC10", "split": "check",
     "claim": "Mount Fuji is located in South Korea.",
     "expected": "REFUTES", "source": "https://whc.unesco.org/en/list/1418/"},
    # Astronomy
    {"id": "AST01S", "category": "Astronomy", "group": "AST01", "split": "selection",
     "claim": "The Milky Way is a barred spiral galaxy.",
     "expected": "SUPPORTS", "source": "https://science.nasa.gov/universe/galaxies/types/"},
    {"id": "AST01R", "category": "Astronomy", "group": "AST01", "split": "selection",
     "claim": "The Milky Way is an elliptical galaxy.",
     "expected": "REFUTES", "source": "https://science.nasa.gov/universe/galaxies/types/"},
    {"id": "AST02S", "category": "Astronomy", "group": "AST02", "split": "selection",
     "claim": "Messier 31 is the Andromeda Galaxy.",
     "expected": "SUPPORTS", "source": "https://science.nasa.gov/mission/hubble/science/explore-the-night-sky/hubble-messier-catalog/messier-31/"},
    {"id": "AST02R", "category": "Astronomy", "group": "AST02", "split": "selection",
     "claim": "Messier 31 is the Whirlpool Galaxy.",
     "expected": "REFUTES", "source": "https://science.nasa.gov/mission/hubble/science/explore-the-night-sky/hubble-messier-catalog/messier-31/"},
    {"id": "AST03S", "category": "Astronomy", "group": "AST03", "split": "selection",
     "claim": "Polaris is located in the constellation Ursa Minor.",
     "expected": "SUPPORTS", "source": "https://science.nasa.gov/solar-system/what-is-the-north-star-and-how-do-you-find-it/"},
    {"id": "AST03R", "category": "Astronomy", "group": "AST03", "split": "selection",
     "claim": "Polaris is located in the constellation Orion.",
     "expected": "REFUTES", "source": "https://science.nasa.gov/solar-system/what-is-the-north-star-and-how-do-you-find-it/"},
    {"id": "AST04S", "category": "Astronomy", "group": "AST04", "split": "check",
     "claim": "The main asteroid belt lies between Mars and Jupiter.",
     "expected": "SUPPORTS", "source": "https://science.nasa.gov/solar-system/asteroids/facts/"},
    {"id": "AST04R", "category": "Astronomy", "group": "AST04", "split": "check",
     "claim": "The main asteroid belt lies between Venus and Earth.",
     "expected": "REFUTES", "source": "https://science.nasa.gov/solar-system/asteroids/facts/"},
    {"id": "AST05S", "category": "Astronomy", "group": "AST05", "split": "check",
     "claim": "Halley's Comet takes approximately 76 years to orbit the Sun.",
     "expected": "SUPPORTS", "source": "https://science.nasa.gov/solar-system/comets/1p-halley/"},
    {"id": "AST05R", "category": "Astronomy", "group": "AST05", "split": "check",
     "claim": "Halley's Comet takes approximately 10 years to orbit the Sun.",
     "expected": "REFUTES", "source": "https://science.nasa.gov/solar-system/comets/1p-halley/"},
    {"id": "AST06S", "category": "Astronomy", "group": "AST06", "split": "selection",
     "claim": "Jupiter's Great Red Spot is a giant atmospheric storm.",
     "expected": "SUPPORTS", "source": "https://science.nasa.gov/asset/hubble/the-great-red-spot-an-ancient-storm-in-jupiters-atmosphere/"},
    {"id": "AST06R", "category": "Astronomy", "group": "AST06", "split": "selection",
     "claim": "Jupiter's Great Red Spot is an impact crater.",
     "expected": "REFUTES", "source": "https://science.nasa.gov/asset/hubble/the-great-red-spot-an-ancient-storm-in-jupiters-atmosphere/"},
    {"id": "AST07S", "category": "Astronomy", "group": "AST07", "split": "selection",
     "claim": "During a solar eclipse seen from Earth, the Moon passes between Earth and the Sun.",
     "expected": "SUPPORTS", "source": "https://science.nasa.gov/eclipses/geometry/"},
    {"id": "AST07R", "category": "Astronomy", "group": "AST07", "split": "selection",
     "claim": "During a solar eclipse seen from Earth, Earth passes between the Moon and the Sun.",
     "expected": "REFUTES", "source": "https://science.nasa.gov/eclipses/geometry/"},
    {"id": "AST08S", "category": "Astronomy", "group": "AST08", "split": "check",
     "claim": "The Hubble Space Telescope was launched in 1990.",
     "expected": "SUPPORTS", "source": "https://www.nasa.gov/image-article/hubble-space-telescope-reaches-orbit/"},
    {"id": "AST08R", "category": "Astronomy", "group": "AST08", "split": "check",
     "claim": "The Hubble Space Telescope was launched in 1980.",
     "expected": "REFUTES", "source": "https://www.nasa.gov/image-article/hubble-space-telescope-reaches-orbit/"},
    {"id": "AST09S", "category": "Astronomy", "group": "AST09", "split": "check",
     "claim": "Ceres is classified as a dwarf planet.",
     "expected": "SUPPORTS", "source": "https://science.nasa.gov/dwarf-planets/ceres/"},
    {"id": "AST09R", "category": "Astronomy", "group": "AST09", "split": "check",
     "claim": "Ceres is classified as a star.",
     "expected": "REFUTES", "source": "https://science.nasa.gov/dwarf-planets/ceres/"},
    {"id": "AST10S", "category": "Astronomy", "group": "AST10", "split": "check",
     "claim": "Mars is the fourth planet from the Sun.",
     "expected": "SUPPORTS", "source": "https://science.nasa.gov/mars/facts/"},
    {"id": "AST10R", "category": "Astronomy", "group": "AST10", "split": "check",
     "claim": "Mars is the sixth planet from the Sun.",
     "expected": "REFUTES", "source": "https://science.nasa.gov/mars/facts/"},
    # Chemistry
    {"id": "CHE01S", "category": "Chemistry", "group": "CHE01", "split": "selection",
     "claim": "Oxygen has atomic number 8.",
     "expected": "SUPPORTS", "source": "https://periodic-table.rsc.org/element/8/oxygen"},
    {"id": "CHE01R", "category": "Chemistry", "group": "CHE01", "split": "selection",
     "claim": "Oxygen has atomic number 12.",
     "expected": "REFUTES", "source": "https://periodic-table.rsc.org/element/8/oxygen"},
    {"id": "CHE02S", "category": "Chemistry", "group": "CHE02", "split": "selection",
     "claim": "The chemical symbol for iron is Fe.",
     "expected": "SUPPORTS", "source": "https://pubchem.ncbi.nlm.nih.gov/element/Fe"},
    {"id": "CHE02R", "category": "Chemistry", "group": "CHE02", "split": "selection",
     "claim": "The chemical symbol for iron is Ir.",
     "expected": "REFUTES", "source": "https://pubchem.ncbi.nlm.nih.gov/element/Fe"},
    {"id": "CHE03S", "category": "Chemistry", "group": "CHE03", "split": "selection",
     "claim": "The chemical symbol for copper is Cu.",
     "expected": "SUPPORTS", "source": "https://pubchem.ncbi.nlm.nih.gov/element/29"},
    {"id": "CHE03R", "category": "Chemistry", "group": "CHE03", "split": "selection",
     "claim": "The chemical symbol for copper is Co.",
     "expected": "REFUTES", "source": "https://pubchem.ncbi.nlm.nih.gov/element/29"},
    {"id": "CHE04S", "category": "Chemistry", "group": "CHE04", "split": "check",
     "claim": "The chemical symbol for chlorine is Cl.",
     "expected": "SUPPORTS", "source": "https://pubchem.ncbi.nlm.nih.gov/element/17"},
    {"id": "CHE04R", "category": "Chemistry", "group": "CHE04", "split": "check",
     "claim": "The chemical symbol for chlorine is Ca.",
     "expected": "REFUTES", "source": "https://pubchem.ncbi.nlm.nih.gov/element/17"},
    {"id": "CHE05S", "category": "Chemistry", "group": "CHE05", "split": "check",
     "claim": "The chemical symbol for helium is He.",
     "expected": "SUPPORTS", "source": "https://pubchem.ncbi.nlm.nih.gov/element/2"},
    {"id": "CHE05R", "category": "Chemistry", "group": "CHE05", "split": "check",
     "claim": "The chemical symbol for helium is H.",
     "expected": "REFUTES", "source": "https://pubchem.ncbi.nlm.nih.gov/element/2"},
    {"id": "CHE06S", "category": "Chemistry", "group": "CHE06", "split": "selection",
     "claim": "Nitrogen makes up approximately 78 percent of Earth's dry atmosphere by volume.",
     "expected": "SUPPORTS", "source": "https://gml.noaa.gov/outreach/carbon_toolkit/"},
    {"id": "CHE06R", "category": "Chemistry", "group": "CHE06", "split": "selection",
     "claim": "Nitrogen makes up approximately 21 percent of Earth's dry atmosphere by volume.",
     "expected": "REFUTES", "source": "https://gml.noaa.gov/outreach/carbon_toolkit/"},
    {"id": "CHE07S", "category": "Chemistry", "group": "CHE07", "split": "selection",
     "claim": "At 20 degrees Celsius and standard atmospheric pressure, mercury is a liquid metal.",
     "expected": "SUPPORTS", "source": "https://periodic-table.rsc.org/element/80/mercury"},
    {"id": "CHE07R", "category": "Chemistry", "group": "CHE07", "split": "selection",
     "claim": "At 20 degrees Celsius and standard atmospheric pressure, mercury is a solid metal.",
     "expected": "REFUTES", "source": "https://periodic-table.rsc.org/element/80/mercury"},
    {"id": "CHE08S", "category": "Chemistry", "group": "CHE08", "split": "check",
     "claim": "Neon has atomic number 10.",
     "expected": "SUPPORTS", "source": "https://periodic-table.rsc.org/element/10/Neon"},
    {"id": "CHE08R", "category": "Chemistry", "group": "CHE08", "split": "check",
     "claim": "Neon has atomic number 20.",
     "expected": "REFUTES", "source": "https://periodic-table.rsc.org/element/10/Neon"},
    {"id": "CHE09S", "category": "Chemistry", "group": "CHE09", "split": "check",
     "claim": "The chemical symbol for lead is Pb.",
     "expected": "SUPPORTS", "source": "https://pubchem.ncbi.nlm.nih.gov/element/Pb"},
    {"id": "CHE09R", "category": "Chemistry", "group": "CHE09", "split": "check",
     "claim": "The chemical symbol for lead is Pd.",
     "expected": "REFUTES", "source": "https://pubchem.ncbi.nlm.nih.gov/element/Pb"},
    {"id": "CHE10S", "category": "Chemistry", "group": "CHE10", "split": "check",
     "claim": "The chemical symbol for aluminum is Al.",
     "expected": "SUPPORTS", "source": "https://pubchem.ncbi.nlm.nih.gov/element/Al"},
    {"id": "CHE10R", "category": "Chemistry", "group": "CHE10", "split": "check",
     "claim": "The chemical symbol for aluminum is Am.",
     "expected": "REFUTES", "source": "https://pubchem.ncbi.nlm.nih.gov/element/Al"},
    # Animals
    {"id": "ANI01S", "category": "Animals", "group": "ANI01", "split": "selection",
     "claim": "Koalas are marsupials.",
     "expected": "SUPPORTS", "source": "https://animals.sandiegozoo.org/animals/koala"},
    {"id": "ANI01R", "category": "Animals", "group": "ANI01", "split": "selection",
     "claim": "Koalas are bears.",
     "expected": "REFUTES", "source": "https://animals.sandiegozoo.org/animals/koala"},
    {"id": "ANI02S", "category": "Animals", "group": "ANI02", "split": "selection",
     "claim": "A giant Pacific octopus normally has eight arms.",
     "expected": "SUPPORTS", "source": "https://www.montereybayaquarium.org/animals/animals-a-to-z/giant-pacific-octopus"},
    {"id": "ANI02R", "category": "Animals", "group": "ANI02", "split": "selection",
     "claim": "A giant Pacific octopus normally has ten arms.",
     "expected": "REFUTES", "source": "https://www.montereybayaquarium.org/animals/animals-a-to-z/giant-pacific-octopus"},
    {"id": "ANI03S", "category": "Animals", "group": "ANI03", "split": "selection",
     "claim": "Platypuses are mammals.",
     "expected": "SUPPORTS", "source": "https://australian.museum/platypus"},
    {"id": "ANI03R", "category": "Animals", "group": "ANI03", "split": "selection",
     "claim": "Platypuses are reptiles.",
     "expected": "REFUTES", "source": "https://australian.museum/platypus"},
    {"id": "ANI04S", "category": "Animals", "group": "ANI04", "split": "check",
     "claim": "Polar bears have black skin beneath their fur.",
     "expected": "SUPPORTS", "source": "https://www.nps.gov/features/bela/arcticAnimalDiscovery/Simple/index.html"},
    {"id": "ANI04R", "category": "Animals", "group": "ANI04", "split": "check",
     "claim": "Polar bears have white skin beneath their fur.",
     "expected": "REFUTES", "source": "https://www.nps.gov/features/bela/arcticAnimalDiscovery/Simple/index.html"},
    {"id": "ANI05S", "category": "Animals", "group": "ANI05", "split": "check",
     "claim": "Giraffes normally have seven neck vertebrae.",
     "expected": "SUPPORTS", "source": "https://vanat.ahc.umn.edu/museum/pages/GiraffeCerVert1a.html"},
    {"id": "ANI05R", "category": "Animals", "group": "ANI05", "split": "check",
     "claim": "Giraffes normally have fourteen neck vertebrae.",
     "expected": "REFUTES", "source": "https://vanat.ahc.umn.edu/museum/pages/GiraffeCerVert1a.html"},
    {"id": "ANI06S", "category": "Animals", "group": "ANI06", "split": "selection",
     "claim": "Rhinoceros horns are made of keratin.",
     "expected": "SUPPORTS", "source": "https://animals.sandiegozoo.org/animals/rhinoceros"},
    {"id": "ANI06R", "category": "Animals", "group": "ANI06", "split": "selection",
     "claim": "Rhinoceros horns are made of ivory.",
     "expected": "REFUTES", "source": "https://animals.sandiegozoo.org/animals/rhinoceros"},
    {"id": "ANI07S", "category": "Animals", "group": "ANI07", "split": "selection",
     "claim": "An elephant's trunk contains no bones.",
     "expected": "SUPPORTS", "source": "https://animals.sandiegozoo.org/animals/elephant"},
    {"id": "ANI07R", "category": "Animals", "group": "ANI07", "split": "selection",
     "claim": "An elephant's trunk contains bones.",
     "expected": "REFUTES", "source": "https://animals.sandiegozoo.org/animals/elephant"},
    {"id": "ANI08S", "category": "Animals", "group": "ANI08", "split": "check",
     "claim": "Crocodiles are reptiles.",
     "expected": "SUPPORTS", "source": "https://animals.sandiegozoo.org/animals/reptiles"},
    {"id": "ANI08R", "category": "Animals", "group": "ANI08", "split": "check",
     "claim": "Crocodiles are mammals.",
     "expected": "REFUTES", "source": "https://animals.sandiegozoo.org/animals/reptiles"},
    {"id": "ANI09S", "category": "Animals", "group": "ANI09", "split": "check",
     "claim": "An adult dromedary camel normally has one hump.",
     "expected": "SUPPORTS", "source": "https://animals.sandiegozoo.org/animals/camel"},
    {"id": "ANI09R", "category": "Animals", "group": "ANI09", "split": "check",
     "claim": "An adult dromedary camel normally has two humps.",
     "expected": "REFUTES", "source": "https://animals.sandiegozoo.org/animals/camel"},
    {"id": "ANI10S", "category": "Animals", "group": "ANI10", "split": "check",
     "claim": "Ostriches are flightless birds.",
     "expected": "SUPPORTS", "source": "https://www.nationalzoo.si.edu/animals/ostrich"},
    {"id": "ANI10R", "category": "Animals", "group": "ANI10", "split": "check",
     "claim": "Ostriches are capable of powered flight.",
     "expected": "REFUTES", "source": "https://www.nationalzoo.si.edu/animals/ostrich"},
    # History and exploration
    {"id": "HIS01S", "category": "History and exploration", "group": "HIS01", "split": "selection",
     "claim": "The Wright brothers made their first successful powered airplane flights in 1903.",
     "expected": "SUPPORTS", "source": "https://www.nps.gov/articles/wrightflyer.htm"},
    {"id": "HIS01R", "category": "History and exploration", "group": "HIS01", "split": "selection",
     "claim": "The Wright brothers made their first successful powered airplane flights in 1803.",
     "expected": "REFUTES", "source": "https://www.nps.gov/articles/wrightflyer.htm"},
    {"id": "HIS02S", "category": "History and exploration", "group": "HIS02", "split": "selection",
     "claim": "Yuri Gagarin became the first human in space in 1961.",
     "expected": "SUPPORTS", "source": "https://www.nasa.gov/history/remembering-yuri-gagarin-50-years-later/"},
    {"id": "HIS02R", "category": "History and exploration", "group": "HIS02", "split": "selection",
     "claim": "Yuri Gagarin became the first human in space in 1951.",
     "expected": "REFUTES", "source": "https://www.nasa.gov/history/remembering-yuri-gagarin-50-years-later/"},
    {"id": "HIS03S", "category": "History and exploration", "group": "HIS03", "split": "check",
     "claim": "Nelson Mandela was inaugurated as president of South Africa in 1994.",
     "expected": "SUPPORTS", "source": "https://www.nelsonmandela.org/on-this-day/entry/nelson-mandela-is-inaugurated-in-pretoria-as-south-africas-first-democratic"},
    {"id": "HIS03R", "category": "History and exploration", "group": "HIS03", "split": "check",
     "claim": "Nelson Mandela was inaugurated as president of South Africa in 1984.",
     "expected": "REFUTES", "source": "https://www.nelsonmandela.org/on-this-day/entry/nelson-mandela-is-inaugurated-in-pretoria-as-south-africas-first-democratic"},
    {"id": "HIS04S", "category": "History and exploration", "group": "HIS04", "split": "check",
     "claim": "The Great Fire of London began in 1666.",
     "expected": "SUPPORTS", "source": "https://www.london-fire.gov.uk/museum/london-fire-brigade-history-and-stories/fires-and-incidents-that-changed-history/the-great-fire-of-london/"},
    {"id": "HIS04R", "category": "History and exploration", "group": "HIS04", "split": "check",
     "claim": "The Great Fire of London began in 1766.",
     "expected": "REFUTES", "source": "https://www.london-fire.gov.uk/museum/london-fire-brigade-history-and-stories/fires-and-incidents-that-changed-history/the-great-fire-of-london/"},
    {"id": "HIS05S", "category": "History and exploration", "group": "HIS05", "split": "check",
     "claim": "RMS Titanic sank in the North Atlantic Ocean.",
     "expected": "SUPPORTS", "source": "https://www.noaa.gov/rms-titanic"},
    {"id": "HIS05R", "category": "History and exploration", "group": "HIS05", "split": "check",
     "claim": "RMS Titanic sank in the Pacific Ocean.",
     "expected": "REFUTES", "source": "https://www.noaa.gov/rms-titanic"},
    {"id": "HIS06S", "category": "History and exploration", "group": "HIS06", "split": "selection",
     "claim": "The United States purchased the Louisiana Territory from France in 1803.",
     "expected": "SUPPORTS", "source": "https://www.archives.gov/publications/prologue/2003/spring/louisiana-purchase.html"},
    {"id": "HIS06R", "category": "History and exploration", "group": "HIS06", "split": "selection",
     "claim": "The United States purchased the Louisiana Territory from France in 1903.",
     "expected": "REFUTES", "source": "https://www.archives.gov/publications/prologue/2003/spring/louisiana-purchase.html"},
    {"id": "HIS07S", "category": "History and exploration", "group": "HIS07", "split": "selection",
     "claim": "The Panama Canal officially opened in 1914.",
     "expected": "SUPPORTS", "source": "https://pancanal.com/en/panama-canal-turns-90/"},
    {"id": "HIS07R", "category": "History and exploration", "group": "HIS07", "split": "selection",
     "claim": "The Panama Canal officially opened in 1814.",
     "expected": "REFUTES", "source": "https://pancanal.com/en/panama-canal-turns-90/"},
    {"id": "HIS08S", "category": "History and exploration", "group": "HIS08", "split": "selection",
     "claim": "Sputnik 1 was launched in 1957.",
     "expected": "SUPPORTS", "source": "https://www.nasa.gov/history/65-years-ago-sputnik-ushers-in-the-space-age/"},
    {"id": "HIS08R", "category": "History and exploration", "group": "HIS08", "split": "selection",
     "claim": "Sputnik 1 was launched in 1967.",
     "expected": "REFUTES", "source": "https://www.nasa.gov/history/65-years-ago-sputnik-ushers-in-the-space-age/"},
    {"id": "HIS09S", "category": "History and exploration", "group": "HIS09", "split": "check",
     "claim": "Valentina Tereshkova became the first woman in space in 1963.",
     "expected": "SUPPORTS", "source": "https://starchild.gsfc.nasa.gov/docs/StarChild/whos_who_level2/tereshkova.html"},
    {"id": "HIS09R", "category": "History and exploration", "group": "HIS09", "split": "check",
     "claim": "Valentina Tereshkova became the first woman in space in 1973.",
     "expected": "REFUTES", "source": "https://starchild.gsfc.nasa.gov/docs/StarChild/whos_who_level2/tereshkova.html"},
    {"id": "HIS10S", "category": "History and exploration", "group": "HIS10", "split": "check",
     "claim": "The Battle of Hastings took place in 1066.",
     "expected": "SUPPORTS", "source": "https://historicengland.org.uk/listing/the-list/list-entry/1000013"},
    {"id": "HIS10R", "category": "History and exploration", "group": "HIS10", "split": "check",
     "claim": "The Battle of Hastings took place in 1166.",
     "expected": "REFUTES", "source": "https://historicengland.org.uk/listing/the-list/list-entry/1000013"},
    # Books and authors
    {"id": "BOO01S", "category": "Books and authors", "group": "BOO01", "split": "selection",
     "claim": "Lewis Carroll wrote Alice's Adventures in Wonderland.",
     "expected": "SUPPORTS", "source": "https://lewiscarrollsociety.org.uk/short-list-of-works/"},
    {"id": "BOO01R", "category": "Books and authors", "group": "BOO01", "split": "selection",
     "claim": "Charles Dickens wrote Alice's Adventures in Wonderland.",
     "expected": "REFUTES", "source": "https://lewiscarrollsociety.org.uk/short-list-of-works/"},
    {"id": "BOO02S", "category": "Books and authors", "group": "BOO02", "split": "selection",
     "claim": "Mary Shelley wrote Frankenstein.",
     "expected": "SUPPORTS", "source": "https://www.nlm.nih.gov/exhibition/frankenstein/index.html"},
    {"id": "BOO02R", "category": "Books and authors", "group": "BOO02", "split": "selection",
     "claim": "Agatha Christie wrote Frankenstein.",
     "expected": "REFUTES", "source": "https://www.nlm.nih.gov/exhibition/frankenstein/index.html"},
    {"id": "BOO03S", "category": "Books and authors", "group": "BOO03", "split": "check",
     "claim": "Mark Twain wrote The Adventures of Tom Sawyer.",
     "expected": "SUPPORTS", "source": "https://americanenglish.state.gov/resources/adventures-tom-sawyer"},
    {"id": "BOO03R", "category": "Books and authors", "group": "BOO03", "split": "check",
     "claim": "Oscar Wilde wrote The Adventures of Tom Sawyer.",
     "expected": "REFUTES", "source": "https://americanenglish.state.gov/resources/adventures-tom-sawyer"},
    {"id": "BOO04S", "category": "Books and authors", "group": "BOO04", "split": "check",
     "claim": "H. G. Wells wrote The War of the Worlds.",
     "expected": "SUPPORTS", "source": "https://blogs.loc.gov/loc/2016/10/lcm-page-from-the-past-war-of-the-worlds/"},
    {"id": "BOO04R", "category": "Books and authors", "group": "BOO04", "split": "check",
     "claim": "Jules Verne wrote The War of the Worlds.",
     "expected": "REFUTES", "source": "https://blogs.loc.gov/loc/2016/10/lcm-page-from-the-past-war-of-the-worlds/"},
    {"id": "BOO05S", "category": "Books and authors", "group": "BOO05", "split": "check",
     "claim": "Bram Stoker wrote Dracula.",
     "expected": "SUPPORTS", "source": "https://www.loc.gov/item/21015337/"},
    {"id": "BOO05R", "category": "Books and authors", "group": "BOO05", "split": "check",
     "claim": "Edgar Allan Poe wrote Dracula.",
     "expected": "REFUTES", "source": "https://www.loc.gov/item/21015337/"},
    {"id": "BOO06S", "category": "Books and authors", "group": "BOO06", "split": "selection",
     "claim": "Leo Tolstoy wrote War and Peace.",
     "expected": "SUPPORTS", "source": "https://www.gutenberg.org/ebooks/2600"},
    {"id": "BOO06R", "category": "Books and authors", "group": "BOO06", "split": "selection",
     "claim": "Fyodor Dostoevsky wrote War and Peace.",
     "expected": "REFUTES", "source": "https://www.gutenberg.org/ebooks/2600"},
    {"id": "BOO07S", "category": "Books and authors", "group": "BOO07", "split": "selection",
     "claim": "Oscar Wilde wrote The Picture of Dorian Gray.",
     "expected": "SUPPORTS", "source": "https://www.loc.gov/item/19012249/"},
    {"id": "BOO07R", "category": "Books and authors", "group": "BOO07", "split": "selection",
     "claim": "Rudyard Kipling wrote The Picture of Dorian Gray.",
     "expected": "REFUTES", "source": "https://www.loc.gov/item/19012249/"},
    {"id": "BOO08S", "category": "Books and authors", "group": "BOO08", "split": "selection",
     "claim": "Emily Bronte wrote Wuthering Heights.",
     "expected": "SUPPORTS", "source": "https://www.bronte.org.uk/the-lives-of-the-brontes/the-bronte-novels"},
    {"id": "BOO08R", "category": "Books and authors", "group": "BOO08", "split": "selection",
     "claim": "Charlotte Bronte wrote Wuthering Heights.",
     "expected": "REFUTES", "source": "https://www.bronte.org.uk/the-lives-of-the-brontes/the-bronte-novels"},
    {"id": "BOO09S", "category": "Books and authors", "group": "BOO09", "split": "check",
     "claim": "Victor Hugo wrote Les Miserables.",
     "expected": "SUPPORTS", "source": "https://www.loc.gov/exhibits/bnf/bnf0006.html"},
    {"id": "BOO09R", "category": "Books and authors", "group": "BOO09", "split": "check",
     "claim": "Alexandre Dumas wrote Les Miserables.",
     "expected": "REFUTES", "source": "https://www.loc.gov/exhibits/bnf/bnf0006.html"},
    {"id": "BOO10S", "category": "Books and authors", "group": "BOO10", "split": "check",
     "claim": "Robert Louis Stevenson wrote Treasure Island.",
     "expected": "SUPPORTS", "source": "https://guides.loc.gov/golden-age-of-piracy/print-resources"},
    {"id": "BOO10R", "category": "Books and authors", "group": "BOO10", "split": "check",
     "claim": "Jules Verne wrote Treasure Island.",
     "expected": "REFUTES", "source": "https://guides.loc.gov/golden-age-of-piracy/print-resources"},
    # Arts and music
    {"id": "ART01S", "category": "Arts and music", "group": "ART01", "split": "selection",
     "claim": "Edvard Munch created The Scream.",
     "expected": "SUPPORTS", "source": "https://www.moma.org/calendar/exhibitions/1305"},
    {"id": "ART01R", "category": "Arts and music", "group": "ART01", "split": "selection",
     "claim": "Claude Monet created The Scream.",
     "expected": "REFUTES", "source": "https://www.moma.org/calendar/exhibitions/1305"},
    {"id": "ART02S", "category": "Arts and music", "group": "ART02", "split": "selection",
     "claim": "Salvador Dali painted The Persistence of Memory.",
     "expected": "SUPPORTS", "source": "https://www.moma.org/collection/works/79018"},
    {"id": "ART02R", "category": "Arts and music", "group": "ART02", "split": "selection",
     "claim": "Pablo Picasso painted The Persistence of Memory.",
     "expected": "REFUTES", "source": "https://www.moma.org/collection/works/79018"},
    {"id": "ART03S", "category": "Arts and music", "group": "ART03", "split": "check",
     "claim": "Ludwig van Beethoven composed Symphony No. 9 in D minor, Op. 125.",
     "expected": "SUPPORTS", "source": "https://www.bso.org/works/symphony-no-9-in-d-minor-op-125-cho"},
    {"id": "ART03R", "category": "Arts and music", "group": "ART03", "split": "check",
     "claim": "Johann Sebastian Bach composed Symphony No. 9 in D minor, Op. 125.",
     "expected": "REFUTES", "source": "https://www.bso.org/works/symphony-no-9-in-d-minor-op-125-cho"},
    {"id": "ART04S", "category": "Arts and music", "group": "ART04", "split": "check",
     "claim": "Wolfgang Amadeus Mozart composed the opera The Magic Flute.",
     "expected": "SUPPORTS", "source": "https://kv.mozarteum.at/en/work/die-zauberflote-7137"},
    {"id": "ART04R", "category": "Arts and music", "group": "ART04", "split": "check",
     "claim": "Giuseppe Verdi composed the opera The Magic Flute.",
     "expected": "REFUTES", "source": "https://kv.mozarteum.at/en/work/die-zauberflote-7137"},
    {"id": "ART05S", "category": "Arts and music", "group": "ART05", "split": "check",
     "claim": "Michelangelo's David is a marble sculpture.",
     "expected": "SUPPORTS", "source": "https://www.galleriaaccademiafirenze.it/en/artworks/david-michelangelo/"},
    {"id": "ART05R", "category": "Arts and music", "group": "ART05", "split": "check",
     "claim": "Michelangelo's David is a wooden sculpture.",
     "expected": "REFUTES", "source": "https://www.galleriaaccademiafirenze.it/en/artworks/david-michelangelo/"},
    {"id": "ART06S", "category": "Arts and music", "group": "ART06", "split": "selection",
     "claim": "Antonio Vivaldi composed The Four Seasons.",
     "expected": "SUPPORTS", "source": "https://www.laphil.com/works/the-four-seasons"},
    {"id": "ART06R", "category": "Arts and music", "group": "ART06", "split": "selection",
     "claim": "Frederic Chopin composed The Four Seasons.",
     "expected": "REFUTES", "source": "https://www.laphil.com/works/the-four-seasons"},
    {"id": "ART07S", "category": "Arts and music", "group": "ART07", "split": "selection",
     "claim": "Pyotr Ilyich Tchaikovsky composed the ballet music for Swan Lake.",
     "expected": "SUPPORTS", "source": "https://www.bso.org/works/tchaikovsky-swan-lake"},
    {"id": "ART07R", "category": "Arts and music", "group": "ART07", "split": "selection",
     "claim": "Wolfgang Amadeus Mozart composed the ballet music for Swan Lake.",
     "expected": "REFUTES", "source": "https://www.bso.org/works/tchaikovsky-swan-lake"},
    {"id": "ART08S", "category": "Arts and music", "group": "ART08", "split": "selection",
     "claim": "Auguste Rodin created the sculpture The Thinker.",
     "expected": "SUPPORTS", "source": "https://www.meudon.musee-rodin.fr/en/musee/collections/oeuvres/penseur-meudon"},
    {"id": "ART08R", "category": "Arts and music", "group": "ART08", "split": "selection",
     "claim": "Donatello created the sculpture The Thinker.",
     "expected": "REFUTES", "source": "https://www.meudon.musee-rodin.fr/en/musee/collections/oeuvres/penseur-meudon"},
    {"id": "ART09S", "category": "Arts and music", "group": "ART09", "split": "check",
     "claim": "Leonardo da Vinci painted The Last Supper in Milan.",
     "expected": "SUPPORTS", "source": "https://cenacolovinciano.org/museo/le-opere/ultima-cena-leonardo-da-vinci-1452-1519/"},
    {"id": "ART09R", "category": "Arts and music", "group": "ART09", "split": "check",
     "claim": "Raphael painted The Last Supper in Milan.",
     "expected": "REFUTES", "source": "https://cenacolovinciano.org/museo/le-opere/ultima-cena-leonardo-da-vinci-1452-1519/"},
    {"id": "ART10S", "category": "Arts and music", "group": "ART10", "split": "check",
     "claim": "Edvard Grieg composed the Peer Gynt Suite No. 1.",
     "expected": "SUPPORTS", "source": "https://www.laphil.com/works/peer-gynt-suite-no-1-op-46"},
    {"id": "ART10R", "category": "Arts and music", "group": "ART10", "split": "check",
     "claim": "Jean Sibelius composed the Peer Gynt Suite No. 1.",
     "expected": "REFUTES", "source": "https://www.laphil.com/works/peer-gynt-suite-no-1-op-46"},
    # Technology and inventions
    {"id": "TEC01S", "category": "Technology and inventions", "group": "TEC01", "split": "selection",
     "claim": "Tim Berners-Lee invented the World Wide Web.",
     "expected": "SUPPORTS", "source": "https://home.cern/science/computing/the-birth-of-the-web"},
    {"id": "TEC01R", "category": "Technology and inventions", "group": "TEC01", "split": "selection",
     "claim": "Bill Gates invented the World Wide Web.",
     "expected": "REFUTES", "source": "https://home.cern/science/computing/the-birth-of-the-web"},
    {"id": "TEC02S", "category": "Technology and inventions", "group": "TEC02", "split": "selection",
     "claim": "IBM introduced the IBM Personal Computer model 5150 in 1981.",
     "expected": "SUPPORTS", "source": "https://www.ibm.com/history/personal-computer"},
    {"id": "TEC02R", "category": "Technology and inventions", "group": "TEC02", "split": "selection",
     "claim": "IBM introduced the IBM Personal Computer model 5150 in 1991.",
     "expected": "REFUTES", "source": "https://www.ibm.com/history/personal-computer"},
    {"id": "TEC03S", "category": "Technology and inventions", "group": "TEC03", "split": "check",
     "claim": "Apple introduced the original Macintosh computer in 1984.",
     "expected": "SUPPORTS", "source": "https://support.apple.com/en-om/112190"},
    {"id": "TEC03R", "category": "Technology and inventions", "group": "TEC03", "split": "check",
     "claim": "Apple introduced the original Macintosh computer in 1994.",
     "expected": "REFUTES", "source": "https://support.apple.com/en-om/112190"},
    {"id": "TEC04S", "category": "Technology and inventions", "group": "TEC04", "split": "check",
     "claim": "Ray Tomlinson chose the @ sign to separate the user name from the destination in network email addresses.",
     "expected": "SUPPORTS", "source": "https://www.internethalloffame.org/inductee/raymond-tomlinson/"},
    {"id": "TEC04R", "category": "Technology and inventions", "group": "TEC04", "split": "check",
     "claim": "Ray Tomlinson chose the # sign to separate the user name from the destination in network email addresses.",
     "expected": "REFUTES", "source": "https://www.internethalloffame.org/inductee/raymond-tomlinson/"},
    {"id": "TEC05S", "category": "Technology and inventions", "group": "TEC05", "split": "check",
     "claim": "Morse code uses both dots and dashes.",
     "expected": "SUPPORTS", "source": "https://www.loc.gov/collections/samuel-morse-papers/articles-and-essays/collection-highlights/invention-of-the-telegraph/"},
    {"id": "TEC05R", "category": "Technology and inventions", "group": "TEC05", "split": "check",
     "claim": "Morse code uses dots but no dashes.",
     "expected": "REFUTES", "source": "https://www.loc.gov/collections/samuel-morse-papers/articles-and-essays/collection-highlights/invention-of-the-telegraph/"},
    {"id": "TEC06S", "category": "Technology and inventions", "group": "TEC06", "split": "selection",
     "claim": "Alexander Graham Bell was granted US Patent No. 174,465 in 1876.",
     "expected": "SUPPORTS", "source": "https://www.loc.gov/collections/alexander-graham-bell-papers/articles-and-essays/timeline/1870-to-1879/"},
    {"id": "TEC06R", "category": "Technology and inventions", "group": "TEC06", "split": "selection",
     "claim": "Alexander Graham Bell was granted US Patent No. 174,465 in 1976.",
     "expected": "REFUTES", "source": "https://www.loc.gov/collections/alexander-graham-bell-papers/articles-and-essays/timeline/1870-to-1879/"},
    {"id": "TEC07S", "category": "Technology and inventions", "group": "TEC07", "split": "selection",
     "claim": "The first working point-contact transistor was demonstrated at Bell Labs in 1947.",
     "expected": "SUPPORTS", "source": "https://www.computerhistory.org/siliconengine/invention-of-the-point-contact-transistor/"},
    {"id": "TEC07R", "category": "Technology and inventions", "group": "TEC07", "split": "selection",
     "claim": "The first working point-contact transistor was demonstrated at Bell Labs in 1967.",
     "expected": "REFUTES", "source": "https://www.computerhistory.org/siliconengine/invention-of-the-point-contact-transistor/"},
    {"id": "TEC08S", "category": "Technology and inventions", "group": "TEC08", "split": "selection",
     "claim": "ENIAC was publicly unveiled in 1946.",
     "expected": "SUPPORTS", "source": "https://almanac.upenn.edu/archive/v42/n18/eniac.html"},
    {"id": "TEC08R", "category": "Technology and inventions", "group": "TEC08", "split": "selection",
     "claim": "ENIAC was publicly unveiled in 1986.",
     "expected": "REFUTES", "source": "https://almanac.upenn.edu/archive/v42/n18/eniac.html"},
    {"id": "TEC09S", "category": "Technology and inventions", "group": "TEC09", "split": "check",
     "claim": "The compact disc was jointly developed by Philips and Sony.",
     "expected": "SUPPORTS", "source": "https://www.philips.com/a-w/about/news/media-library/20190101-Philips-Compact-Disc-player-1982-1983.html"},
    {"id": "TEC09R", "category": "Technology and inventions", "group": "TEC09", "split": "check",
     "claim": "The compact disc was jointly developed by Philips and Nintendo.",
     "expected": "REFUTES", "source": "https://www.philips.com/a-w/about/news/media-library/20190101-Philips-Compact-Disc-player-1982-1983.html"},
    {"id": "TEC10S", "category": "Technology and inventions", "group": "TEC10", "split": "check",
     "claim": "Erno Rubik invented the Rubik's Cube in 1974.",
     "expected": "SUPPORTS", "source": "https://rubiks.com/history"},
    {"id": "TEC10R", "category": "Technology and inventions", "group": "TEC10", "split": "check",
     "claim": "Erno Rubik invented the Rubik's Cube in 1984.",
     "expected": "REFUTES", "source": "https://rubiks.com/history"},
    # Sports and games
    {"id": "SPO01S", "category": "Sports and games", "group": "SPO01", "split": "selection",
     "claim": "A standard chessboard has 64 squares.",
     "expected": "SUPPORTS", "source": "https://handbook.fide.com/chapter/E012023"},
    {"id": "SPO01R", "category": "Sports and games", "group": "SPO01", "split": "selection",
     "claim": "A standard chessboard has 100 squares.",
     "expected": "REFUTES", "source": "https://handbook.fide.com/chapter/E012023"},
    {"id": "SPO02S", "category": "Sports and games", "group": "SPO02", "split": "selection",
     "claim": "The first men's FIFA World Cup was held in Uruguay in 1930.",
     "expected": "SUPPORTS", "source": "https://inside.fifa.com/news/world-first-for-uruguay-2053846"},
    {"id": "SPO02R", "category": "Sports and games", "group": "SPO02", "split": "selection",
     "claim": "The first men's FIFA World Cup was held in Italy in 1930.",
     "expected": "REFUTES", "source": "https://inside.fifa.com/news/world-first-for-uruguay-2053846"},
    {"id": "SPO03S", "category": "Sports and games", "group": "SPO03", "split": "check",
     "claim": "James Naismith invented basketball in 1891.",
     "expected": "SUPPORTS", "source": "https://www.fiba.basketball/en/news/basketballs-first-game-was-125-years-ago-today"},
    {"id": "SPO03R", "category": "Sports and games", "group": "SPO03", "split": "check",
     "claim": "James Naismith invented basketball in 1791.",
     "expected": "REFUTES", "source": "https://www.fiba.basketball/en/news/basketballs-first-game-was-125-years-ago-today"},
    {"id": "SPO04S", "category": "Sports and games", "group": "SPO04", "split": "check",
     "claim": "The first modern Olympic Games were held in Athens in 1896.",
     "expected": "SUPPORTS", "source": "https://olympicgames.culture.gov.gr/en/revival.html"},
    {"id": "SPO04R", "category": "Sports and games", "group": "SPO04", "split": "check",
     "claim": "The first modern Olympic Games were held in Rome in 1896.",
     "expected": "REFUTES", "source": "https://olympicgames.culture.gov.gr/en/revival.html"},
    {"id": "SPO05S", "category": "Sports and games", "group": "SPO05", "split": "check",
     "claim": "Under association football rules, the maximum number of players per team on the field at once is eleven.",
     "expected": "SUPPORTS", "source": "https://www.theifab.com/laws/latest/the-players/"},
    {"id": "SPO05R", "category": "Sports and games", "group": "SPO05", "split": "check",
     "claim": "Under association football rules, the maximum number of players per team on the field at once is thirteen.",
     "expected": "REFUTES", "source": "https://www.theifab.com/laws/latest/the-players/"},
    {"id": "SPO06S", "category": "Sports and games", "group": "SPO06", "split": "selection",
     "claim": "The Wimbledon tennis championships are played on grass courts.",
     "expected": "SUPPORTS", "source": "https://www.wimbledon.com/en_GB/about/grass_courts"},
    {"id": "SPO06R", "category": "Sports and games", "group": "SPO06", "split": "selection",
     "claim": "The Wimbledon tennis championships are played on clay courts.",
     "expected": "REFUTES", "source": "https://www.wimbledon.com/en_GB/about/grass_courts"},
    {"id": "SPO07S", "category": "Sports and games", "group": "SPO07", "split": "selection",
     "claim": "Badminton is played with a shuttlecock rather than a ball.",
     "expected": "SUPPORTS", "source": "https://library.olympics.com/Default/digitalCollection/DigitalCollectionAttachmentDownloadHandler.ashx?documentId=207321&parentDocumentId=206701&skipCopyright=true&skipWatermark=true"},
    {"id": "SPO07R", "category": "Sports and games", "group": "SPO07", "split": "selection",
     "claim": "Badminton is played with a ball rather than a shuttlecock.",
     "expected": "REFUTES", "source": "https://library.olympics.com/Default/digitalCollection/DigitalCollectionAttachmentDownloadHandler.ashx?documentId=207321&parentDocumentId=206701&skipCopyright=true&skipWatermark=true"},
    {"id": "SPO08S", "category": "Sports and games", "group": "SPO08", "split": "selection",
     "claim": "The standard marathon distance is 42.195 kilometres.",
     "expected": "SUPPORTS", "source": "https://worldathletics.org/disciplines/road-running/marathon"},
    {"id": "SPO08R", "category": "Sports and games", "group": "SPO08", "split": "selection",
     "claim": "The standard marathon distance is 21.0975 kilometres.",
     "expected": "REFUTES", "source": "https://worldathletics.org/disciplines/road-running/marathon"},
    {"id": "SPO09S", "category": "Sports and games", "group": "SPO09", "split": "check",
     "claim": "Under standard rugby union rules, the maximum number of players per team on the field at once is fifteen.",
     "expected": "SUPPORTS", "source": "https://passport.world.rugby/laws-of-the-game/laws-by-number/3-team/"},
    {"id": "SPO09R", "category": "Sports and games", "group": "SPO09", "split": "check",
     "claim": "Under standard rugby union rules, the maximum number of players per team on the field at once is thirteen.",
     "expected": "REFUTES", "source": "https://passport.world.rugby/laws-of-the-game/laws-by-number/3-team/"},
    {"id": "SPO10S", "category": "Sports and games", "group": "SPO10", "split": "check",
     "claim": "A standard fifteen-red snooker ball set contains 22 balls, including the white cue ball.",
     "expected": "SUPPORTS", "source": "https://wpbsa.com/wp-content/uploads/2198_WPBSA-Rulebook-2024-25.pdf"},
    {"id": "SPO10R", "category": "Sports and games", "group": "SPO10", "split": "check",
     "claim": "A standard fifteen-red snooker ball set contains 16 balls, including the white cue ball.",
     "expected": "REFUTES", "source": "https://wpbsa.com/wp-content/uploads/2198_WPBSA-Rulebook-2024-25.pdf"},
]
# END EDITABLE CLAIMS


def canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_claims(claims: list[dict]) -> None:
    # Reject a broken/customised manifest before importing or loading models.
    total = EXPECTED_CLAIM_COUNT
    per_label, per_split = total // 2, total // 2
    per_split_label = total // 4
    per_category = total // EXPECTED_CATEGORY_COUNT
    per_category_split_label = total // (4 * EXPECTED_CATEGORY_COUNT)
    if total % (4 * EXPECTED_CATEGORY_COUNT):
        raise ValueError("Expected count must allow balanced categories, splits and labels.")
    if not isinstance(claims, list) or len(claims) != total:
        raise ValueError(f"CLAIMS must contain exactly {total} editable claim records.")
    required = {"id", "category", "group", "split", "claim", "expected", "source"}
    ids, texts = set(), set()
    groups = defaultdict(list)
    for index, row in enumerate(claims, 1):
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"Claim {index} must contain {sorted(required)}.")
        for field in required:
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Claim {index}: {field} must be a nonempty string.")
            if row[field] != row[field].strip():
                raise ValueError(f"Claim {index}: trim surrounding whitespace in {field}.")
        if row["id"] in ids:
            raise ValueError(f"Duplicate claim ID: {row['id']}")
        ids.add(row["id"])
        normalised = " ".join(row["claim"].casefold().split())
        if normalised in texts:
            raise ValueError(f"Duplicate claim text: {row['id']}")
        texts.add(normalised)
        if len(row["claim"]) > 4000:
            raise ValueError(f"Claim {row['id']} exceeds the application's character limit.")
        if row["expected"] not in LABELS or row["split"] not in {"selection", "check"}:
            raise ValueError(f"Claim {row['id']} has an invalid expected label or split.")
        source = urlparse(row["source"])
        if (source.scheme != "https" or not source.hostname or source.username
                or source.password or any(c.isspace() for c in row["source"])):
            raise ValueError(f"Claim {row['id']} needs a valid HTTPS source URL.")
        groups[row["group"]].append(row)
    if Counter(row["expected"] for row in claims) != Counter(SUPPORTS=per_label, REFUTES=per_label):
        raise ValueError(f"CLAIMS must have {per_label} SUPPORTS and {per_label} REFUTES records.")
    category_counts = Counter(row["category"] for row in claims)
    if len(category_counts) != EXPECTED_CATEGORY_COUNT or set(category_counts.values()) != {per_category}:
        raise ValueError(f"CLAIMS must have {EXPECTED_CATEGORY_COUNT} categories with {per_category} records each.")
    for split in ("selection", "check"):
        subset = [row for row in claims if row["split"] == split]
        if (len(subset) != per_split or Counter(r["expected"] for r in subset)
                != Counter(SUPPORTS=per_split_label, REFUTES=per_split_label)):
            raise ValueError(f"The {split} split must have {per_split} records, {per_split_label} per label.")
        if set(r["category"] for r in subset) != set(category_counts):
            raise ValueError(f"The {split} split must include all {EXPECTED_CATEGORY_COUNT} categories.")
        for category in category_counts:
            labels = Counter(r["expected"] for r in subset if r["category"] == category)
            if labels != Counter(SUPPORTS=per_category_split_label, REFUTES=per_category_split_label):
                raise ValueError(f"Category {category} in {split} needs {per_category_split_label} records per label.")
    for name, rows in groups.items():
        if (len(rows) != 2 or len({row["split"] for row in rows}) != 1
                or len({row["category"] for row in rows}) != 1
                or Counter(r["expected"] for r in rows) != Counter(SUPPORTS=1, REFUTES=1)):
            raise ValueError(f"Group {name} must be a same-category/split SUPPORTS/REFUTES pair.")


def describe_times(values: list[float]) -> dict:
    return {"mean_ms": statistics.mean(values), "p50_ms": statistics.median(values),
            "minimum_ms": min(values), "maximum_ms": max(values)}


def compute_metrics(rows: list[dict], k: int) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    truth = [row["expected"] for row in rows]
    predictions = [row["predictions"][str(k)] for row in rows]
    predicted = [prediction["label"] for prediction in predictions]
    matrix = confusion_matrix(truth, predicted, labels=LABELS).tolist()
    report = classification_report(truth, predicted, labels=LABELS, output_dict=True, zero_division=0)
    correct = sum(a == b for a, b in zip(truth, predicted))
    classifications = [p["classification_ms"] for p in predictions]
    retrievals = [row["retrieval_ms"] for row in rows]
    return {
        "k": k, "correct": correct, "total": len(rows), "incorrect": len(rows) - correct,
        "accuracy": float(accuracy_score(truth, predicted)),
        "agreement_pct": 100 * correct / len(rows),
        "macro_f1": float(f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)),
        "supports_correct": matrix[0][0], "supports_total": sum(matrix[0]),
        "refutes_correct": matrix[1][1], "refutes_total": sum(matrix[1]),
        "supports_recall": report["SUPPORTS"]["recall"],
        "refutes_recall": report["REFUTES"]["recall"],
        "predicted_labels": {label: predicted.count(label) for label in LABELS},
        "high_confidence_errors": sum(p["confidence"] >= .99 and p["label"] != expected
                                      for p, expected in zip(predictions, truth)),
        "labels": LABELS, "confusion_matrix": matrix, "classification_report": report,
        "timings": {
            "retrieval_top10": describe_times(retrievals),
            "classification": describe_times(classifications),
            "retrieval_top10_plus_classification": describe_times(
                [r + c for r, c in zip(retrievals, classifications)]),
        },
        "chunks": {"total": sum(p["chunks"] for p in predictions),
                   "mean": statistics.mean(p["chunks"] for p in predictions),
                   "maximum": max(p["chunks"] for p in predictions)},
    }


def selection_key(metrics: dict) -> tuple:
    # Exact rational scores prevent rounded or floating-point tie decisions.
    cm = metrics["confusion_matrix"]
    f1s = []
    for i in range(2):
        tp = cm[i][i]
        denominator = sum(cm[i]) + sum(row[i] for row in cm)
        f1s.append(Fraction(2 * tp, denominator) if denominator else Fraction(0))
    return sum(f1s) / 2, Fraction(metrics["correct"], metrics["total"]), -metrics["k"]


def choose_k(selection_metrics: list[dict]) -> dict:
    winner = max(selection_metrics, key=selection_key)
    exact_f1, exact_accuracy, _ = selection_key(winner)
    return {
        "selected_k": winner["k"], "selected_using": f"selection split only ({winner['total']} claims)",
        "rule": "highest exact macro F1, then highest exact accuracy, then smallest k",
        "selection_macro_f1": winner["macro_f1"], "selection_accuracy": winner["accuracy"],
        "exact_selection_macro_f1": str(exact_f1),
        "exact_selection_accuracy": str(exact_accuracy),
    }


def source_files() -> list[Path]:
    return [Path(__file__).resolve(), *[ROOT / name for name in (
        "Backend/settings.py", "Backend/retrieval.py", "Backend/bert_model.py",
        "Backend/verification.py", "Python Scripts/prepare_evidence.py",
        "Retrievers/common.py", "Retrievers/hybrid_retriever.py",
        "Retrievers/embedding_retriever.py", "Retrievers/bm25_retriever.py",
    )]]


def file_manifest(paths: list[Path]) -> dict:
    result = {}
    for path in sorted(set(paths)):
        if not path.is_file():
            raise FileNotFoundError(f"Required local evaluation asset is missing: {path}")
        try:
            name = path.relative_to(ROOT).as_posix()
        except ValueError:
            name = path.as_posix()
        result[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def save_figures(result: dict, directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay

    # Match the existing evaluation figures: plain Matplotlib and Blues matrices.
    colours = {"selection": "#1f77b4", "check": "#ff7f0e", "all": "#555555"}
    total = len(result["claims"])
    names = {"selection": f"Selection ({result['metrics']['selection'][0]['total']})",
             "check": f"Reserved check ({result['metrics']['check'][0]['total']})",
             "all": f"All ({total})"}
    k = result["recommendation"]["selected_k"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, field, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro F1")):
        for split in ("selection", "check", "all"):
            ax.plot(KS, [100*m[field] for m in result["metrics"][split]], marker="o",
                    label=names[split], color=colours[split])
        ax.axvline(k, color="#777777", linestyle=":", linewidth=1)
        ax.set(title=title, xlabel="Retrieved passages (top k)", ylabel=f"{title} (%)",
               xticks=KS, ylim=(0, 100))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"{total} general claims: evidence count comparison")
    fig.text(.5, .01, f"k={k} selected on selection claims only; all-claim results are descriptive.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(directory / "performance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, split in zip(axes, ("selection", "check", "all")):
        for field, label, colour in (("supports_recall", "SUPPORTS recall", "#1f77b4"),
                                      ("refutes_recall", "REFUTES recall", "#d62728")):
            ax.plot(KS, [100*m[field] for m in result["metrics"][split]], marker="o",
                    label=label, color=colour)
        ax.axvline(k, color="#777777", linestyle=":", linewidth=1)
        ax.set(title=names[split], xlabel="Retrieved passages (top k)",
               xticks=KS, ylim=(0, 100))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Recall (%)")
    fig.suptitle("Correctly supported true claims and correctly refuted false claims")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(directory / "class_recall.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    for ax, m in zip(axes.flat, result["metrics"]["all"]):
        display = ConfusionMatrixDisplay(np.asarray(m["confusion_matrix"]), display_labels=LABELS)
        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d",
                     im_kw={"vmin": 0, "vmax": max(m['supports_total'], m['refutes_total'])})
        ax.set_title(f"Top {m['k']} — {m['correct']}/{m['total']} correct")
        ax.tick_params(axis="both", labelsize=8)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle(f"All {total} claims: confusion matrix at each evidence count")
    label_counts = Counter(row["expected"] for row in result["claims"])
    fig.text(.5, .012, "Rows = expected; columns = predicted. "
             + "; ".join(f"{label}: {label_counts[label]} expected claims" for label in LABELS)
             + ". These full-set matrices did not select k.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(directory / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def validate_prediction(prediction: dict) -> None:
    probabilities = prediction["probabilities"]
    if (prediction["label"] not in LABELS or set(probabilities) != set(LABELS)
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6)
            or not math.isclose(prediction["confidence"], probabilities[prediction["label"]], abs_tol=1e-7)
            or prediction["confidence"] < max(probabilities.values())
            or not isinstance(prediction["chunks"], int) or prediction["chunks"] < 1):
        raise RuntimeError("BERT returned an invalid binary prediction.")


def configure_determinism(seed: int, torch, np) -> None:
    # Require deterministic inference before either model is constructed.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def without_timings(value):
    # Retain every metric/category value except measured wall-clock timings.
    if isinstance(value, dict):
        return {key: without_timings(item) for key, item in value.items() if key != "timings"}
    if isinstance(value, list):
        return [without_timings(item) for item in value]
    return value


def verify_reproducibility(result: dict, previous: dict, previous_sha256: str) -> dict:
    # Fail closed on changed inputs, assets, device, evidence or predictions.
    identity_keys = (
        "case_manifest_sha256", "source_files", "local_assets", "embedding_model",
        "checkpoint", "corpus_passages", "versions", "inference_settings", "device_identity",
        "device", "embedding_device", "ks", "labels", "api_calls", "new_bert_predictions",
        "reused_predictions", "app_top_k_unchanged",
    )
    for key in identity_keys:
        if key not in result or key not in previous or result[key] != previous[key]:
            raise RuntimeError(f"Repeat-run comparison failed: {key} differs or is missing.")
    if len(result["claims"]) != len(previous.get("claims", [])):
        raise RuntimeError("Repeat-run comparison failed: claim count differs.")
    maximum_delta = 0.0
    for current, prior in zip(result["claims"], previous["claims"]):
        excluded = {"retrieval_ms", "predictions"}
        current_input_evidence = {key: value for key, value in current.items() if key not in excluded}
        prior_input_evidence = {key: value for key, value in prior.items() if key not in excluded}
        if current_input_evidence != prior_input_evidence:
            raise RuntimeError(f"Repeat-run comparison failed: {current['id']} claim/order/evidence ranking differs.")
        if set(current["predictions"]) != set(prior.get("predictions", {})):
            raise RuntimeError(f"Repeat-run comparison failed: {current['id']} prediction counts differ.")
        for k in KS:
            prediction, old = current["predictions"][str(k)], prior["predictions"][str(k)]
            validate_prediction(prediction)
            validate_prediction(old)
            ignored = {"classification_ms", "confidence", "probabilities"}
            if ({key: value for key, value in prediction.items() if key not in ignored}
                    != {key: value for key, value in old.items() if key not in ignored}):
                raise RuntimeError(f"Repeat-run comparison failed: {current['id']} k={k} prediction differs.")
            pairs = [(prediction["confidence"], old["confidence"]), *[
                (prediction["probabilities"][label], old["probabilities"][label]) for label in LABELS]]
            for a, b in pairs:
                delta = abs(a - b)
                maximum_delta = max(maximum_delta, delta)
                if delta > PROBABILITY_TOLERANCE:
                    raise RuntimeError(f"Repeat-run comparison failed: {current['id']} k={k} "
                                       f"confidence/probabilities differ by {delta:.9g}.")
    for key in ("metrics", "categories"):
        if without_timings(result[key]) != without_timings(previous.get(key)):
            raise RuntimeError(f"Repeat-run comparison failed: timing-free {key} differ.")
    if result["recommendation"] != previous.get("recommendation"):
        raise RuntimeError("Repeat-run comparison failed: selected k or selection metrics differ.")
    return {
        "status": "verified", "compared_results_sha256": previous_sha256,
        "compared_result_created_at_utc": previous.get("created_at_utc"),
        "compared_manifest_sha256": previous["case_manifest_sha256"],
        "claims_compared": len(result["claims"]),
        "predictions_compared": len(result["claims"]) * len(KS),
        "probability_absolute_tolerance": PROBABILITY_TOLERANCE,
        "maximum_observed_probability_or_confidence_delta": maximum_delta,
        "provenance_settings_and_device_comparison": "exact",
        "claim_order_and_evidence_ranking_comparison": "exact",
        "predicted_labels_and_chunk_counts_comparison": "exact",
        "aggregate_and_category_metrics_comparison": "exact excluding timings",
        "timing_fields_compared": False,
    }


def run(args) -> dict:
    validate_claims(CLAIMS)
    # Deep-copy the exact editable manifest before any model imports or inference.
    claims = json.loads(canonical_json(CLAIMS))
    manifest_sha256 = hashlib.sha256(canonical_json(claims)).hexdigest()
    before_sources = file_manifest(source_files())
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    # cuBLAS reads this on CUDA initialization; set it before torch/model imports.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from Backend.settings import Settings
    from Backend.retrieval import EvidenceRetriever
    from Backend.bert_model import BertClassifier
    import torch
    import numpy as np

    configure_determinism(args.seed, torch, np)

    settings = Settings(device=args.device)
    issues = settings.asset_issues()
    if issues:
        raise FileNotFoundError(f"Required local assets are missing: {', '.join(issues)}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested, but CUDA is unavailable.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    assets = [settings.index_dir / "passages.jsonl", settings.index_dir / "embeddings.faiss",
              *[p for p in (settings.index_dir / "bm25").rglob("*") if p.is_file()],
              *[p for p in settings.checkpoint_dir.rglob("*") if p.is_file()]]
    total = len(claims)
    split_counts = Counter(row["split"] for row in claims)
    label_counts = Counter(row["expected"] for row in claims)
    print(f"Validated {total} claims: {split_counts['selection']} selection / "
          f"{split_counts['check']} reserved check; {label_counts['SUPPORTS']} SUPPORTS / "
          f"{label_counts['REFUTES']} REFUTES.", flush=True)
    print("Hashing local source, checkpoint and retrieval assets before inference...", flush=True)
    before_assets = file_manifest(assets)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    tick = perf_counter()
    retriever = EvidenceRetriever(settings)
    # The original retriever auto-selects a device; make the CLI apply to its queries too.
    retriever.retriever.emb.model.to(device)
    retriever.retriever.emb.model.eval()
    sync()
    retrieval_load = perf_counter() - tick
    tick = perf_counter()
    classifier = BertClassifier(settings)
    sync()
    bert_load = perf_counter() - tick
    print(f"Loaded retriever ({retrieval_load:.2f}s) and BERT ({bert_load:.2f}s) on {device}.", flush=True)

    # Record the cached, pinned MiniLM snapshot without network access or downloads.
    from huggingface_hub.constants import HF_HUB_CACHE
    from embedding_retriever import MODEL, MODEL_REVISION
    snapshot = Path(HF_HUB_CACHE) / ("models--" + MODEL.replace("/", "--")) / "snapshots" / MODEL_REVISION
    if not snapshot.is_dir():
        raise FileNotFoundError("The pinned MiniLM snapshot is not available in the local Hugging Face cache.")
    embedding_assets = [p for p in snapshot.rglob("*") if p.is_file()]
    before_embedding = file_manifest(embedding_assets)
    rows = []
    metrics = {}
    frozen_choice = None
    started = perf_counter()
    for split in ("selection", "check"):
        subset = [row for row in claims if row["split"] == split]
        for index, claim in enumerate(subset, 1):
            sync()
            tick = perf_counter()
            hits = retriever.search(claim["claim"], k=10)
            sync()
            retrieval_ms = (perf_counter() - tick) * 1000
            if len(hits) != 10:
                raise RuntimeError(f"{claim['id']}: expected ten retrieved passages, got {len(hits)}.")
            predictions = {}
            for k in KS:
                rendered = "\n".join(dict.fromkeys(hit["rendered"] for hit in hits[:k]))
                sync()
                tick = perf_counter()
                prediction = classifier.predict(claim["claim"], rendered)
                sync()
                classification_ms = (perf_counter() - tick) * 1000
                validate_prediction(prediction)
                predictions[str(k)] = {**prediction, "classification_ms": classification_ms,
                    "matches_expected": prediction["label"] == claim["expected"],
                    "passages_selected": k, "unique_rendered_passages": len(dict.fromkeys(
                        hit["rendered"] for hit in hits[:k]))}
            rows.append({**claim, "retrieval_ms": retrieval_ms, "evidence_top10": hits,
                         "predictions": predictions})
            if index % 10 == 0:
                print(f"{split}: completed {index}/{len(subset)} claims (ten passage counts each).", flush=True)
        split_rows = [row for row in rows if row["split"] == split]
        metrics[split] = [compute_metrics(split_rows, k) for k in KS]
        if split == "selection":
            frozen_choice = choose_k(metrics[split])
            print(f"Selection locked top k={frozen_choice['selected_k']}; beginning reserved check claims.", flush=True)
    inference_seconds = perf_counter() - started
    # Return rows in editable manifest order rather than split processing order.
    by_id = {row["id"]: row for row in rows}
    rows = [by_id[row["id"]] for row in claims]
    metrics["all"] = [compute_metrics(rows, k) for k in KS]
    if choose_k(metrics["selection"]) != frozen_choice:
        raise RuntimeError("Selection decision changed after the reserved check split.")
    print("Checking source and local asset hashes after inference...", flush=True)
    if file_manifest(source_files()) != before_sources:
        raise RuntimeError("Evaluation or application source changed during this run; results not published.")
    if file_manifest(assets) != before_assets or file_manifest(embedding_assets) != before_embedding:
        raise RuntimeError("A local model/index asset changed during this run; results not published.")
    current = datetime.now(timezone.utc)
    versions = {}
    for package in ("torch", "transformers", "sentence-transformers", "bm25s", "faiss-cpu",
                    "numpy", "scipy", "scikit-learn", "matplotlib"):
        versions[package] = importlib.metadata.version(package)
    categories = [{"name": name, "metrics": {
        split: [compute_metrics([r for r in rows if r["category"] == name
                                 and (split == "all" or r["split"] == split)], k) for k in KS]
        for split in ("selection", "check", "all")}}
        for name in dict.fromkeys(row["category"] for row in claims)]
    return {
        "created_at_utc": current.isoformat(),
        "client_date": current.astimezone(ZoneInfo(CLIENT_TIMEZONE)).date().isoformat(),
        "client_timezone": CLIENT_TIMEZONE,
        "test_name": f"{total} editable general claims: top 1 through top 10",
        "ks": KS, "labels": LABELS, "device": str(classifier.device),
        "embedding_device": str(retriever.retriever.emb.model.device),
        "api_calls": 0, "new_bert_predictions": total * len(KS),
        "reused_predictions": 0, "app_top_k_unchanged": settings.top_k,
        "case_manifest_sha256": manifest_sha256,
        "source_files": before_sources, "local_assets": before_assets,
        "embedding_model": {"name": MODEL, "revision": MODEL_REVISION, "files": before_embedding},
        "checkpoint": settings.checkpoint_dir.relative_to(ROOT).as_posix(),
        "corpus_passages": len(retriever.retriever.passages),
        "versions": {"python": platform.python_version(), **versions},
        "inference_settings": {
            "seed": args.seed, "python_random_seed": args.seed, "numpy_seed": args.seed,
            "torch_seed": torch.initial_seed(), "cuda_seed_all": args.seed,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
            "max_length": settings.max_length, "chunk_batch_size": settings.chunk_batch_size,
            "bert_dtype": str(classifier.model.dtype),
            "bert_attention_implementation": classifier.model.config._attn_implementation,
            "bert_training": classifier.model.training,
            "embedding_training": retriever.retriever.emb.model.training,
            "processing_order": [row["id"] for split in ("selection", "check")
                                 for row in claims if row["split"] == split],
            "prefix_order": KS,
        },
        "device_identity": {
            "platform": platform.platform(), "machine": platform.machine(),
            "torch_cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
            "gpu_compute_capability": list(torch.cuda.get_device_capability()) if device == "cuda" else None,
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None,
        },
        "reproducibility_verification": {"status": "not_requested"},
        "load_timings_seconds": {"retrieval": retrieval_load, "bert": bert_load},
        "inference_seconds": inference_seconds,
        "recommendation": frozen_choice,
        "notes": [
            f"{total} source-labelled general claims across {len(categories)} categories; "
            f"{label_counts['SUPPORTS']} SUPPORTS and {label_counts['REFUTES']} REFUTES. "
            "Labels/sources are fixed in the editable Python CLAIMS list before inference.",
            "Related true/false examples share a group and remain in one split. "
            f"The selection split contains {split_counts['selection']} claims "
            f"({split_counts['selection']//2} per label); {split_counts['check']} claims "
            f"({split_counts['check']//2} per label) form a reserved check split. "
            "Every category is balanced by label in both splits.",
            "Every claim is independently retrieved once using the existing hybrid BM25/MiniLM "
            "ranking. Top k uses the first k passages of that claim-specific top-ten ranking.",
            "Rendered passages are deduplicated and newline-joined exactly as in the application. "
            "The existing BERT adapter repeats claims across lossless evidence chunks, averages "
            f"raw chunk logits and then applies softmax. All {total * len(KS):,} predictions are newly computed.",
            "k is frozen using selection-only exact macro F1, then accuracy, then smaller k "
            "before any check predictions are computed. Check and combined results never reselect k.",
            "This expanded diagnostic reuses earlier evaluated claims and adds further claims after "
            "observing earlier pipeline results. It is not an unbiased test set or an independently "
            "held-out FEVER benchmark. Training-data overlap and semantic evidence sufficiency "
            "have not been assessed; the reserved split is only an internal check.",
            "A correct label without sufficient evidence is not successful verification. Confidence "
            "is BERT's binary model confidence, not a probability that a claim is true. "
            "Explanations, citation grounding and NOT ENOUGH INFO behavior are not evaluated.",
            "Per-k timing uses measured top-ten retrieval plus BERT classification for that prefix; "
            "it is not a separately measured top-k retrieval endpoint. Load times are separate. "
            "No warm-up predictions are excluded; early calls can include kernel/caching overhead. "
            "CUDA is synchronized around measured operations when used.",
            "All inference uses local, read-only model/index assets with Hugging Face offline mode. "
            "No OpenAI/Claude calls, key reads, training, index rebuilding or app setting changes "
            f"are made. The application default at run time is {settings.top_k}; "
            "users may choose one through ten.",
        ],
        "metrics": metrics, "categories": categories, "claims": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Compare top 1-10 retrieved passages on an editable 200-claim diagnostic.\n'
            '\n'
            'Run from the repository root::\n'
            '\n'
            '    .\\.venv\\Scripts\\python.exe "Python Scripts\\evaluate_general_claims.py"\n'
            '\n'
            'Edit CLAIMS below to repeat this experiment. Related true/false examples must\n'
            'share a group and split. The selection split chooses k before the check split\n'
            'is evaluated; check results never change that selection. This runner uses only\n'
            'local retrieval/BERT assets and does not load an LLM client or read API keys.\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Device for BERT and embedding queries (default: auto).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fixed Python/NumPy/PyTorch/CUDA seed (default: 42).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for JSON and three image files (default: Evaluation/general_claims_topk).")
    parser.add_argument("--compare-to", type=Path,
                        help="Verify against a prior results.json; mismatch stops publication. "
                             "The prior JSON is loaded before inference, so the output path is safe.")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1.")
    # Snapshot before inference/output writes, including when comparing to this output path.
    previous = None
    previous_sha256 = None
    if args.compare_to is not None:
        raw_previous = args.compare_to.resolve().read_bytes()
        previous = json.loads(raw_previous)
        previous_sha256 = hashlib.sha256(raw_previous).hexdigest()
    result = run(args)
    if previous is not None:
        result["reproducibility_verification"] = verify_reproducibility(
            result, previous, previous_sha256)
        print("Repeat-run verification passed for all inputs, evidence, predictions and timing-free metrics.", flush=True)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    save_figures(result, directory)
    (directory / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for m in result["metrics"]["all"]:
        print(f"k={m['k']:2}: {m['correct']:3}/{m['total']}; accuracy={100*m['accuracy']:.2f}%; "
              f"macro F1={m['macro_f1']:.4f}", flush=True)
    print(f"Selection-only candidate default: {result['recommendation']['selected_k']}; "
          f"application default unchanged ({result['app_top_k_unchanged']}).", flush=True)
    print(f"Saved JSON and three image files in {directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
