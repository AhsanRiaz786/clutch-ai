"""
classifier/dataset.py
Clutch.ai — Training Dataset

Classes:
    0 — technical_question   : CS technical interview questions
    1 — personal_behavioral  : Personal/behavioral interview questions (tell me about yourself, STAR, experience)
    2 — noise                : Non-interview random speech to skip
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Class 0 — Technical Questions
# ---------------------------------------------------------------------------
TECHNICAL_QUESTIONS = [
    # Data Structures
    "What is the difference between a stack and a queue?",
    "Explain how a hash table works and what happens during a collision.",
    "What is a binary search tree and what are its properties?",
    "How does a linked list differ from an array?",
    "What is a heap data structure and when would you use it?",
    "Explain the difference between a min-heap and a max-heap.",
    "What is a trie and what is it used for?",
    "How does a balanced binary tree maintain its balance?",
    "What is the difference between a graph and a tree?",
    "How is a circular linked list different from a regular linked list?",
    "What is a deque and how is it different from a queue?",
    "Explain the concept of a priority queue.",
    "What is an adjacency matrix versus an adjacency list?",
    "What is the purpose of a sentinel node in a linked list?",
    "How does a skip list work?",
    "What is the difference between a B-tree and a binary search tree?",
    "How does a hash map handle resizing?",
    "What is the time complexity of inserting into a balanced BST?",
    "Explain the concept of a disjoint set or union-find.",
    "What is a Bloom filter?",
    # Algorithms
    "What is Big O notation and why does it matter?",
    "Explain the concept of recursion with an example.",
    "How does merge sort work?",
    "What is the time complexity of quicksort in the average case?",
    "Explain the difference between breadth-first and depth-first search.",
    "What is dynamic programming and when would you use it?",
    "What is memoization?",
    "Explain Dijkstra's algorithm and its time complexity.",
    "How does binary search work and what is its time complexity?",
    "What is the difference between greedy algorithms and dynamic programming?",
    "Explain the concept of divide and conquer.",
    "What is backtracking and give an example of when to use it.",
    "How does a topological sort work?",
    "What is the Floyd-Warshall algorithm used for?",
    "What is amortized time complexity?",
    "Explain the concept of tail recursion.",
    "What is the difference between DFS and BFS in terms of space complexity?",
    "How does Kruskal's algorithm work?",
    "What is a sliding window technique?",
    "How does the two-pointer technique work?",
    # Networking
    "What is the difference between TCP and UDP?",
    "How does HTTPS work?",
    "What is the CAP theorem?",
    "What is load balancing and why is it important?",
    "Explain what a REST API is and how it works.",
    "What is DNS and how does it work?",
    "How does a three-way handshake work in TCP?",
    "What is NAT and why is it used?",
    "What is a CDN and how does it improve performance?",
    "Explain the OSI model and its layers.",
    "What is the difference between IPv4 and IPv6?",
    "How does a firewall work?",
    "What is a reverse proxy?",
    "What is the difference between HTTP/1.1 and HTTP/2?",
    "How does TLS encryption work?",
    # Operating Systems
    "What is the difference between a process and a thread?",
    "What is a deadlock and how do you prevent it?",
    "What is a semaphore and when would you use one?",
    "Explain how virtual memory works.",
    "What is context switching?",
    "What is a race condition?",
    "What is the difference between preemptive and non-preemptive scheduling?",
    "Explain the producer-consumer problem.",
    "What is a mutex lock?",
    "What is memory paging versus segmentation?",
    "What is thrashing in operating systems?",
    "How does the CPU scheduler work?",
    # Databases
    "Explain the difference between SQL and NoSQL databases.",
    "What is a foreign key in a relational database?",
    "How does indexing improve database query performance?",
    "What is ACID in the context of databases?",
    "What is database normalization?",
    "What is the difference between INNER JOIN and LEFT JOIN?",
    "What is database sharding?",
    "What is the N+1 query problem?",
    "What is a database transaction?",
    "What is eventual consistency?",
    "What is a distributed database?",
    "How does a B-tree index work in databases?",
    # OOP
    "What is polymorphism in object-oriented programming?",
    "What is the difference between an abstract class and an interface?",
    "Explain encapsulation in OOP.",
    "What is inheritance and what are its advantages?",
    "What is method overriding versus method overloading?",
    "What is the SOLID principle?",
    "Explain the singleton design pattern.",
    "What is the factory design pattern?",
    "What is the observer pattern?",
    "What is the decorator pattern?",
    # Languages & System Design
    "How does garbage collection work in Python?",
    "What is a closure in JavaScript?",
    "What is the GIL in Python?",
    "What is a generator in Python?",
    "What is async and await and how does it work?",
    "What is a microservice architecture?",
    "What is a message queue and when would you use one?",
    "How does a cache work and what are common eviction policies?",
    "What is a rate limiter?",
    "Explain the difference between horizontal and vertical scaling.",
    "What is the DRY principle?",
    "What is type inference?",
    "How does a compiler differ from an interpreter?",
    "What is the size of int in C?",
    "How does a CPU execute instructions?",
]

# ---------------------------------------------------------------------------
# Class 1 — Personal / Behavioral (interview-relevant personal questions)
# ---------------------------------------------------------------------------
PERSONAL_BEHAVIORAL = [
    # Intro / Tell me about yourself
    "Tell me about yourself.",
    "Walk me through your resume.",
    "Introduce yourself.",
    "Give me a quick intro.",
    "Can you introduce yourself and your background?",
    "Tell me a bit about your background.",
    "Who are you and what have you worked on?",
    # Experience
    "What is your experience with software development?",
    "Tell me about your work experience.",
    "What projects have you worked on?",
    "What is the most interesting project you have built?",
    "Describe your most challenging project.",
    "Tell me about a project you are proud of.",
    "What technologies have you worked with?",
    "Tell me about your internship experience.",
    "What did you learn from your last project?",
    "Have you worked on any machine learning or AI projects?",
    "Tell me about a time you worked in a team.",
    "What open source contributions have you made?",
    "Describe a project where you had to learn something new quickly.",
    # Behavioral STAR
    "Tell me about a time you faced a difficult challenge and how you handled it.",
    "Describe a situation where you had to meet a tight deadline.",
    "Tell me about a time you disagreed with a teammate.",
    "Give me an example of a time you showed leadership.",
    "Tell me about a time you failed and what you learned from it.",
    "Describe a time you had to work with a difficult person.",
    "Tell me about a time you went above and beyond.",
    "Describe a situation where you had to debug a hard problem.",
    "Tell me about a time you improved a process or system.",
    "Give an example of a time you had to explain a technical concept to a non-technical person.",
    "Tell me about a time you had multiple competing priorities.",
    "Describe a time you received critical feedback and how you responded.",
    # Strengths / Weaknesses
    "What are your greatest strengths?",
    "What is your biggest weakness?",
    "What makes you a good software engineer?",
    "What sets you apart from other candidates?",
    "How would your teammates describe you?",
    "What is something you are still working on improving?",
    # Motivation / Goals
    "Why did you choose computer science?",
    "Why are you interested in this role?",
    "Why do you want to work at this company?",
    "Where do you see yourself in five years?",
    "What are your career goals?",
    "What motivates you as an engineer?",
    "Why are you looking for a new opportunity?",
    "What kind of problems excite you the most?",
    "What type of engineering work do you enjoy most?",
    "What do you want to learn in your next role?",
    # Education
    "Tell me about your education.",
    "What did you study in university?",
    "What relevant courses have you taken?",
    "Tell me about your final year project or thesis.",
    "What is your GPA and academic background?",
    "Have you done any research during your degree?",
    # Soft skills / Culture fit
    "How do you handle stress and pressure?",
    "How do you prioritize tasks when you have many things to do?",
    "Do you prefer working alone or in a team?",
    "How do you keep up with the latest technology trends?",
    "How do you approach learning a new technology?",
    "Describe your ideal work environment.",
    "How do you handle feedback and criticism?",
    "What do you do when you are stuck on a problem?",
    # Salary / Logistics
    "What are your salary expectations?",
    "When can you start?",
    "Are you open to relocation?",
    "Do you have any questions for us?",
    "What are you looking for in your next position?",
]

# ---------------------------------------------------------------------------
# Class 2 — Noise (non-interview random speech to skip)
# ---------------------------------------------------------------------------
NOISE = [
    # Random background conversation
    "What is the capital of France?",
    "What is the capital of Hungary?",
    "How is the weather today?",
    "The weather has been great lately.",
    "Did you see the football match last night?",
    "What did you have for lunch today?",
    "Can you pass me the coffee?",
    "I need to check my phone for a second.",
    "Let me get some water.",
    "Hold on I'll be right back.",
    "Can you hear me okay?",
    "Is the mic working?",
    "Let me adjust my camera.",
    "Can you see my screen?",
    "The internet connection seems slow.",
    "I think there is a lag.",
    # Filler sounds
    "Um okay.",
    "Yeah right.",
    "Sure thing.",
    "Alright.",
    "Okay.",
    "Uh huh.",
    "Hmm.",
    "Let me think.",
    "Good.",
    "I see.",
    "Okay great.",
    "Thank you.",
    "Thanks.",
    "Yes.",
    "No problem.",
    # Random statements
    "I am going to grab some coffee.",
    "I need a minute.",
    "Just a moment.",
    "Give me a second.",
    "Okay let me check something.",
    "I will be right back.",
    "Can you repeat that?",
    "Sorry I didn't catch that.",
    "Please go ahead.",
    "All right.",
    # Non-interview trivia or off-topic
    "What is the boiling point of water?",
    "Who won the World Cup?",
    "What is the population of Pakistan?",
    "Who invented the telephone?",
    "What year did World War 2 end?",
    "What is two plus two?",
    "Are you gate?",
    "You are a blessing.",
    "Art is done go go go.",
    "All the CPUs work.",
    "This is the time complexity.",
    "Report close.",
    "Mode.",
    "Good care.",
    "So look after you guys.",
]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "technical_question":  0,
    "personal_behavioral": 1,
    "noise":               2,
}
CLASS_NAMES = list(LABEL_MAP.keys())


def build_dataset():
    """
    Embeds all examples and returns (X, y) numpy arrays.
    Also prints class distribution.
    """
    from sentence_transformers import SentenceTransformer

    all_texts  = []
    all_labels = []

    for text in TECHNICAL_QUESTIONS:
        all_texts.append(text)
        all_labels.append(LABEL_MAP["technical_question"])

    for text in PERSONAL_BEHAVIORAL:
        all_texts.append(text)
        all_labels.append(LABEL_MAP["personal_behavioral"])

    for text in NOISE:
        all_texts.append(text)
        all_labels.append(LABEL_MAP["noise"])

    counts = {c: all_labels.count(i) for c, i in LABEL_MAP.items()}
    print(f"[DATASET] Class distribution: {' | '.join(f'{k}={v}' for k,v in counts.items())} | Total={len(all_texts)}")

    embedder_name = "all-MiniLM-L6-v2"
    print(f"[DATASET] Loading embedder: {embedder_name} ...")
    embedder = SentenceTransformer(embedder_name)

    X = embedder.encode(all_texts, show_progress_bar=True)
    y = np.array(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[DATASET] Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test, embedder_name


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class InterviewDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
