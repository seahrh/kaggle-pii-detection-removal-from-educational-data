import random
from typing import List, NamedTuple, Optional, Set

import jinja2
from faker import Faker

__all__ = ["Prompter"]


def labels(
    text: str,
    tokens: List[str],
    street_address: str,
    student_name: str,
    username: str,
    personal_url: str,
) -> List[str]:
    res = ["O"] * len(tokens)
    # s1 = text.lower()
    # s2 = street_address.lower()
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
        2. Give an example that involved someone named {{ username }}
        3. Give an example that involved {{ personal_url }} [/INST]
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
        self.fake = Faker(locales)
        tset: Set[str] = set() if street_addresses is None else street_addresses
        while len(tset) < target_size:
            tset.add(str(self.fake.address()).replace("\n", " "))
        self.street_addresses: List[str] = list(tset)
        tset = set() if student_names is None else student_names
        while len(tset) < target_size:
            tset.add(f"{self.fake.first_name()} {self.fake.last_name()}")
        self.student_names: List[str] = list(tset)
        tset = set() if usernames is None else usernames
        while len(tset) < target_size:
            tset.add(self.username())
        self.usernames: List[str] = list(tset)
        tset = set() if personal_urls is None else personal_urls
        while len(tset) < target_size:
            tset.add(self.personal_url())
        self.personal_urls: List[str] = list(tset)

    def username(self) -> str:
        separators: List[str] = ["_", "__", "-", "--", ".", "", ""]
        seps: List[str] = [random.choice(separators)] * 6
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
        head: str = random.choice(extra_words)
        if head == "numeric":
            head = str(random.randint(1, 9999))
        mid: str = random.choice(extra_words)
        if mid == "numeric":
            mid = str(random.randint(1, 9999))
        tail: str = random.choice(extra_words)
        if tail == "numeric":
            tail = str(random.randint(1, 9999))
        first_name: str = self.fake.first_name()
        last_name: str = self.fake.last_name()
        res = f"{seps[0]}{head}{seps[1]}{first_name}{seps[2]}{mid}{seps[3]}{last_name}{seps[4]}{tail}{seps[5]}"
        res = res.strip(".")
        p = random.random()
        if p >= 0.95:
            return res.upper()
        if p >= 0.20:
            return res.lower()
        return res

    def personal_url(self) -> str:
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
        username = self.username().replace(".", "")
        res = f"{domain_name}{username}"
        return res

    def prompt(self) -> Result:
        domain: str = random.choice(self.DOMAINS)
        essay_question: str = random.choice(self.ESSAY_QUESTIONS)
        place: str = random.choice(self.PLACES)
        street_address: str = random.choice(self.street_addresses)
        student_name: str = random.choice(self.student_names)
        username: str = random.choice(self.usernames)
        personal_url: str = random.choice(self.personal_urls)
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
