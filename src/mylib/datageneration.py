import random
from typing import List, NamedTuple, Optional, Set

import jinja2
from faker import Faker

__all__ = ["Prompter"]


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
        "Defense Industry and Military Strategy",
        "Maritime Industry and Port Management",
        "Biotechnology and Medical Devices",
        "Environmental Conservation and Management",
        "Government Policy and Administration",
        "Architecture and Urban Design",
        "Industrial Manufacturing and Technology Innovation",
        "Public Administration and Governance",
        "Maritime Engineering and Naval Architecture",
        "Medical Research and Pharmaceuticals",
        "Sustainable Urban Development",
        "Civil Engineering and Infrastructure",
        "Environmental Planning and Sustainability",
        "Healthcare Policy and Reform",
        "Transportation Planning and Logistics",
    ]

    ESSAY_QUESTIONS: List[str] = [
        "how does reflection contribute to preventing misrepresentation in the design thinking process? Discuss how the design thinking approach aids in gathering insights and tackling challenges. Explore the role of visualization, mapping, and other visual tools in accurately representing information and ideas.",
        "how can visualization tools aid stakeholders in a design thinking approach? Discuss the role of graphic representation and reflection in creating visual insights, and explore how such methods can be applied within an organization to generate innovative ideas.",
        "how can the use of storytelling within the design thinking process overcome challenges and enhance understanding across the whole organisation? Discuss the role of storytelling in conveying experiences, engaging audiences, and fostering insight.",
        "how can the integration of a learning launch tool within the design thinking process enhance the development and testing of ideas, prototypes, and products? Discuss how this approach can address the launch challenge, involve stakeholders, and facilitate learning throughout the project development process.",
        "how can the integration of a mind mapping tool within the design thinking process enhance creative idea generation and insight development? Discuss the role of mind mapping in research, idea refinement, and fostering collaboration among stakeholders.",
        "how does the design thinking process foster innovation and creativity within various activities? Explore the role of brainstorming, prototyping, and research in developing innovative ideas and approaches. Discuss how the design thinking process differs from traditional problem-solving methods.",
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
        separators: List[str] = ["", "_", "__", "-", "--", "."]
        seps: List[str] = [random.choice(separators)] * 6
        extra_words: List[str] = ["", "the", "official", "iam", "xxx", "numeric"]
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
            "blogspot.me/",
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
