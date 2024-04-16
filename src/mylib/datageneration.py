import random
from typing import List, NamedTuple, Optional, Set

import jinja2
from faker import Faker
from spacy.tokens import Token

__all__ = ["Prompter", "labels"]


def labels(
    text: str,
    tokens: List[Token],
    street_address: str,
    student_name: str,
    username: str,
    personal_url: str,
) -> List[str]:
    print(f"v={street_address}")
    res = ["O"] * len(tokens)
    s1 = text.lower()

    def _apply(label: str, value: str) -> None:
        s2 = value.lower()
        i, j = 0, 0
        while i < len(text):
            i = s1.find(s2, i)
            if i < 0:
                break
            beginning = True
            while j < len(tokens):
                if i <= tokens[j].idx < i + len(s2):
                    if beginning:
                        res[j] = f"B-{label}"
                        beginning = False
                    else:
                        res[j] = f"I-{label}"
                j += 1

    _apply(label="STREET_ADDRESS", value=street_address)
    _apply(label="NAME_STUDENT", value=student_name)
    _apply(label="USERNAME", value=username)
    _apply(label="URL_PERSONAL", value=personal_url)
    return res


class Prompter:

    class Result(NamedTuple):
        prompt: str
        street_address: str
        student_name: str
        username: str
        personal_url: str

    DOMAINS: List[str] = [
        "Education Policy and Reform",
        "Renewable Energy Solutions",
        "Community Development and Social Services",
        "Technology Innovation and Development",
        "Economic Development and Policy",
        "Labour Policy and Workforce Management",
        "Disaster Management and Emergency Response",
        "Aerospace and Aviation Industry",
        "Defense Industry and National Security",
        "Maritime Industry and Port Management",
        "Biotechnology and Medical Devices",
        "Environmental Conservation and Management",
        "Government Policy and Administration",
        "Architecture and Urban Design",
        "Industrial Manufacturing and Technology Innovation",
        "Public Administration and Governance",
        "Naval Architecture and Shipbuilding",
        "Medical Research and Pharmaceuticals",
        "Sustainable Urban Development",
        "Civil Engineering and Infrastructure",
        "Environmental Planning and Sustainability",
        "Healthcare Policy and Reform",
        "Civil Society and Social Equity",
        "Social Welfare and Support Services",
        "Emerging Technologies and Digital Transformation",
        "Human Resources Management and Organizational Development",
        "Crisis Response and Humanitarian Aid",
        "Space Exploration and Satellite Technology",
        "Ocean Conservation and Coastal Management",
        "Genetic Engineering and Biomedical Research",
        "Biodiversity Preservation and Restoration",
        "Policy Analysis and Government Reform",
        "Sustainable Architecture and Urban Planning",
        "Advanced Manufacturing and Industry 4.0",
        "Policy Implementation and Public Service Delivery",
        "Public Service Delivery and Efficiency",
        "Marine Engineering and Offshore Structures",
        "Pharmaceutical Development and Clinical Trials",
        "Smart Cities and Urban Resilience",
        "Climate Change Adaptation and Resilience Planning",
        "Public Health Initiatives and Healthcare Systems",
        "Logistics and Supply Chain Management",
        "Government Transparency and Accountability",
        "Talent Development and Organizational Behavior",
        "Social Justice and Community Welfare",
        "Workforce Development and Talent Management",
        "Community Empowerment and Social Development",
    ]

    ESSAY_QUESTIONS: List[str] = [
        "how does reflection contribute in tackling challenges and preventing misrepresentation in the design thinking process?",
        "how can visualization tools foster innovation and collaboration in a design thinking approach?",
        "how can storytelling be used within the design thinking process to foster innovation and collaboration?",
        "how can the integration of a learning launch tool within the design thinking process enhance the development and testing of ideas, prototypes, and products?",
        "how can the integration of a mind mapping tool within the design thinking process generate creative ideas and insights?",
        "how does brainstorming and prototyping within the design thinking process foster creativity and teamwork?",
    ]

    PLACES: List[str] = [
        "home",
        "school",
        "office",
        "village",
        "town",
        "city plaza",
        "nature park",
        "shopping mall",
        "train station",
        "bus stop",
        "subway station",
    ]

    def __init__(
        self,
        target_size: int,
        street_addresses: Optional[Set[str]] = None,
        student_names: Optional[Set[str]] = None,
        usernames: Optional[Set[str]] = None,
        personal_urls: Optional[Set[str]] = None,
        seed: int = 31,
    ):
        Faker.seed(seed)
        random.seed(seed)
        environment = jinja2.Environment()
        self.template = environment.from_string(
            """[INST] Write a short essay on the following topic:
        In {{ domain }}, {{ essay_question }}
        Intersperse the following information in different parts of the essay:
        1. Give an example that occurred near {{ street_address }}, at a {{ place }} frequented by {{ student_name }}
        2. Give an example citing: {{ username }}
        3. Give an example citing: {{ personal_url }} [/INST]
        """
        )
        locales = [
            "en_US",
            "en_IN",
            "en_PH",
            "en_GB",
            "de_DE",
            "it_IT",
            "es_ES",
            "da_DK",
            "fr_FR",
            "vi_VN",
            "id_ID",
        ]
        fake = Faker(locales)
        tset: Set[str] = set() if street_addresses is None else street_addresses
        while len(tset) < target_size:
            tset.add(str(fake.address()).replace("\n", " "))
        self.street_addresses: List[str] = list(tset)
        tset = set() if student_names is None else student_names
        while len(tset) < target_size:
            tset.add(f"{fake.first_name()} {fake.last_name()}")
        self.student_names: List[str] = list(tset)

    def username(self, name: str) -> str:
        seps: List[str] = ["_", "__", "-", "--", ".", "", "", "", ""]
        random.shuffle(seps)
        extra_words: List[str] = [
            "the",
            "official",
            "iam",
            "xxx",
            "numeric",
            "",
            "",
            "",
            "",
        ]
        random.shuffle(extra_words)
        head: str = extra_words[0]
        if head == "numeric":
            head = str(random.randint(1, 9999))
        mid: str = extra_words[1]
        if mid == "numeric":
            mid = str(random.randint(1, 9999))
        tail: str = extra_words[2]
        if tail == "numeric":
            tail = str(random.randint(1, 9999))
        parts = name.split()
        first_name: str = parts[0]
        last_name: str = parts[-1]
        res = f"{seps[0]}{head}{seps[1]}{first_name}{seps[2]}{mid}{seps[3]}{last_name}{seps[4]}{tail}{seps[5]}"
        res = res.strip(".")
        p = random.random()
        if p >= 0.95:
            return res.upper()
        if p >= 0.20:
            return res.lower()
        return res

    def personal_url(self, name: str) -> str:
        domain_names = [
            "linkedin.com/",
            "youtube.com/",
            "instagram.com/",
            "smarturl.it/",
            "github.com/",
            "facebook.com/",
            "twitter.com/@",
            "tiktok.com/@",
            "medium.com/@",
            "blogspot.me/@",
            "pinterest.com/",
            "snapchat.com/",
            "reddit.com/u/",
            "tumblr.com/",
            "whatsapp.com/",
            "wechat.com/",
            "telegram.org/",
            "flickr.com/",
            "quora.com/",
            "vimeo.com/",
            "discord.com/",
            "twitch.tv/",
            "meetup.com/",
            "soundcloud.com/",
            "vine.co/",
            "periscope.tv/",
            "dribbble.com/",
            "behance.net/",
            "VK.com/",
            "sinaweibo.com/",
            "renren.com/",
            "foursquare.com/",
            "myspace.com/",
        ]
        domain_name = random.choice(domain_names)
        username = self.username(name=name)
        username = username.replace(".", "")
        username = username.replace("_", "")
        res = f"{domain_name}{username}"
        return res

    def prompt(self) -> Result:
        domain: str = random.choice(self.DOMAINS)
        essay_question: str = random.choice(self.ESSAY_QUESTIONS)
        place: str = random.choice(self.PLACES)
        street_address: str = random.choice(self.street_addresses)
        student_name: str = random.choice(self.student_names)
        username: str = self.username(name=student_name)
        personal_url: str = self.personal_url(name=student_name)
        return self.Result(
            prompt=self.template.render(
                domain=domain,
                essay_question=essay_question,
                street_address=street_address,
                place=place,
                student_name=student_name,
                username=username,
                personal_url=personal_url,
            ),
            street_address=street_address,
            student_name=student_name,
            username=username,
            personal_url=personal_url,
        )
