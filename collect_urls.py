import json
import random
import requests
import wikipediaapi

FIXED_COUNT = 3
RANDOM_COUNT = 5

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='HybridRAGAssignmentBot/1.0'
)


# ---------------- RANDOM PAGE FETCH ----------------

def get_random_title():
    url = "https://en.wikipedia.org/api/rest_v1/page/random/title"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        return None

    data = response.json()

    return data["items"][0]["title"]


# ---------------- VALIDATION ----------------

def is_valid_page(title):
    page = wiki.page(title)

    if not page.exists():
        return False

    word_count = len(page.text.split())

    if word_count < 200:
        return False

    return True


# ---------------- COLLECTOR ----------------

def collect_urls(target_count, existing_urls=set()):

    collected = set(existing_urls)

    print(f"Collecting {target_count} URLs...")

    while len(collected) < target_count:

        title = get_random_title()

        if title is None:
            continue

        url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")

        if url in collected:
            continue

        try:
            if is_valid_page(title):
                collected.add(url)
                print(f"Collected {len(collected)}/{target_count}")

        except Exception:
            continue

    return list(collected)


# ---------------- FIXED URL COLLECTION ----------------

try:
    with open("data/fixed_urls.json") as f:
        fixed_urls = set(json.load(f))
        print("Existing fixed URLs found:", len(fixed_urls))
except:
    fixed_urls = set()

if len(fixed_urls) < FIXED_COUNT:
    fixed_urls = collect_urls(FIXED_COUNT, fixed_urls)

with open("data/fixed_urls.json", "w") as f:
    json.dump(list(fixed_urls), f, indent=2)

print("Fixed URLs saved:", len(fixed_urls))


# ---------------- RANDOM URL COLLECTION ----------------

random_urls = collect_urls(RANDOM_COUNT)

with open("data/random_urls.json", "w") as f:
    json.dump(random_urls, f, indent=2)

print("Random URLs saved:", len(random_urls))


print("\nDONE")
print("Fixed:", len(fixed_urls))
print("Random:", len(random_urls))
