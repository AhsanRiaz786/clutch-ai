"""
classifier/dataset.py
Clutch.ai — Training Dataset

Classes:
    0 — technical_question   : CS technical interview questions (interviewer asking candidate)
    1 — personal_behavioral  : Personal/behavioral/STAR interview questions
    2 — noise                : Non-question speech — filler, candidate's own answers, background
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Class 0 — Technical Questions (~200 examples)
# Phrasing variety matters: interviewers ask differently every time.
# ---------------------------------------------------------------------------
TECHNICAL_QUESTIONS = [
    # ── Data Structures ──────────────────────────────────────────────────────
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
    "Can you walk me through how a stack works?",
    "What would you use a queue for in practice?",
    "How would you implement a stack using only queues?",
    "What is an LRU cache and how would you implement one?",
    "What is a doubly linked list?",
    "How do you detect a cycle in a linked list?",
    "What is the difference between a tree and a graph?",
    "How does a red-black tree differ from an AVL tree?",
    "What data structure would you use to implement a browser's back button?",
    "How would you find the middle of a linked list?",
    # ── Algorithms ───────────────────────────────────────────────────────────
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
    "Can you explain how quicksort partitions the array?",
    "What is the difference between stable and unstable sorting algorithms?",
    "How would you find all permutations of a string?",
    "What is the Knapsack problem?",
    "How does the A-star search algorithm work?",
    "What is the difference between Prim's and Kruskal's algorithm?",
    "How do you detect a cycle in a directed graph?",
    "What is a topological ordering and when is it applicable?",
    "How would you implement binary search on a rotated sorted array?",
    "What is the difference between best-case and worst-case complexity?",
    # ── Networking ───────────────────────────────────────────────────────────
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
    "What is the difference between authentication and authorization?",
    "How does OAuth work?",
    "What is a WebSocket and when would you use it over HTTP?",
    "What is CORS and why does it exist?",
    "How does a browser resolve a URL to an IP address?",
    # ── Operating Systems ─────────────────────────────────────────────────────
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
    "What is the difference between a mutex and a semaphore?",
    "How does inter-process communication work?",
    "What is a system call?",
    "What is the difference between user space and kernel space?",
    "Explain how fork works in Unix.",
    "What is a zombie process?",
    # ── Databases ─────────────────────────────────────────────────────────────
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
    "What is the difference between a primary key and a unique key?",
    "What is a stored procedure?",
    "What is the difference between optimistic and pessimistic locking?",
    "What is connection pooling?",
    "How does a database handle concurrent writes?",
    "What is a composite index?",
    # ── OOP & Design Patterns ─────────────────────────────────────────────────
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
    "What is the difference between composition and inheritance?",
    "What is the strategy design pattern?",
    "What is the command pattern?",
    "What is dependency injection?",
    "How does the MVC pattern work?",
    # ── Languages / Runtime ───────────────────────────────────────────────────
    "How does garbage collection work in Python?",
    "What is a closure in JavaScript?",
    "What is the GIL in Python?",
    "What is a generator in Python?",
    "What is async and await and how does it work?",
    "What is the event loop in JavaScript?",
    "What is the difference between pass by value and pass by reference?",
    "What is type inference?",
    "How does a compiler differ from an interpreter?",
    "What is the difference between a compiled and interpreted language?",
    "What is memory management in C++?",
    "What is a smart pointer in C++?",
    "What is the difference between stack and heap memory?",
    "What is a dangling pointer?",
    # ── System Design ─────────────────────────────────────────────────────────
    "What is a microservice architecture?",
    "What is a message queue and when would you use one?",
    "How does a cache work and what are common eviction policies?",
    "What is a rate limiter?",
    "Explain the difference between horizontal and vertical scaling.",
    "How would you design a URL shortener?",
    "How would you design a notification system?",
    "What is an API gateway?",
    "What is service discovery?",
    "What is the difference between synchronous and asynchronous communication?",
    "What is a circuit breaker pattern?",
    "How would you handle distributed transactions?",
    "What is a saga pattern in microservices?",
    "How does consistent hashing work?",
    "What is the difference between monolithic and microservice architecture?",
    # ── ML / AI (since it's a CS/AI interview context) ───────────────────────
    "What is the difference between supervised and unsupervised learning?",
    "Explain gradient descent.",
    "What is overfitting and how do you prevent it?",
    "What is the bias-variance tradeoff?",
    "What is a transformer model?",
    "What is the difference between a CNN and an RNN?",
    "What is backpropagation?",
    "What is a loss function?",
    "What is regularization in machine learning?",
    "What is cross-entropy loss?",
    "What is the attention mechanism?",
    "How does a random forest work?",
    "What is the curse of dimensionality?",
]

# ---------------------------------------------------------------------------
# Class 1 — Personal / Behavioral (~130 examples)
# Includes intros, STAR questions, motivation, experience, education, culture.
# ---------------------------------------------------------------------------
PERSONAL_BEHAVIORAL = [
    # ── Introduction ──────────────────────────────────────────────────────────
    "Tell me about yourself.",
    "Walk me through your resume.",
    "Introduce yourself.",
    "Give me a quick intro.",
    "Can you introduce yourself and your background?",
    "Tell me a bit about your background.",
    "Who are you and what have you worked on?",
    "Start by telling me about yourself.",
    "How would you describe yourself as an engineer?",
    "Give me your elevator pitch.",
    # ── Experience & Projects ──────────────────────────────────────────────────
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
    "What was your role in your last project?",
    "Tell me about the tech stack you are most comfortable with.",
    "What was the biggest technical challenge you solved?",
    "Tell me about something you built from scratch.",
    "What is the most complex system you have worked on?",
    "Describe a time you had to pick up a new technology fast.",
    "Tell me about your final year project.",
    # ── Behavioral STAR ───────────────────────────────────────────────────────
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
    "Give me an example of when you took initiative.",
    "Tell me about a time you mentored someone.",
    "Describe a time when you had to make a decision without all the information.",
    "Tell me about a conflict you had with a team member and how you resolved it.",
    "Describe a time you had to push back on a bad idea.",
    "Tell me about a time a project did not go as planned.",
    "Give an example of a time you influenced a technical decision.",
    "Tell me about a time you had to learn something completely new.",
    "Describe a situation where you had to meet a very tight timeline.",
    "Tell me about the most impactful thing you have done at your last job or project.",
    "Describe a time you worked under pressure.",
    "Have you ever had to deliver bad news to a client or manager?",
    "Tell me about a time you had to balance technical debt with feature delivery.",
    # ── Strengths & Weaknesses ─────────────────────────────────────────────────
    "What are your greatest strengths?",
    "What is your biggest weakness?",
    "What makes you a good software engineer?",
    "What sets you apart from other candidates?",
    "How would your teammates describe you?",
    "What is something you are still working on improving?",
    "What is your superpower as an engineer?",
    "What is one area where you need to grow?",
    "How do you handle working on tasks you find boring?",
    # ── Motivation & Goals ─────────────────────────────────────────────────────
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
    "Why are you leaving your current role?",
    "What does your ideal job look like?",
    "What kind of company culture do you thrive in?",
    "What are you passionate about in tech?",
    "Where do you see the industry heading?",
    # ── Education ─────────────────────────────────────────────────────────────
    "Tell me about your education.",
    "What did you study in university?",
    "What relevant courses have you taken?",
    "Tell me about your final year project or thesis.",
    "What is your GPA and academic background?",
    "Have you done any research during your degree?",
    "What was your favorite course in university?",
    "Did you have any relevant coursework?",
    "How has your education prepared you for this role?",
    # ── Soft Skills & Culture Fit ──────────────────────────────────────────────
    "How do you handle stress and pressure?",
    "How do you prioritize tasks when you have many things to do?",
    "Do you prefer working alone or in a team?",
    "How do you keep up with the latest technology trends?",
    "How do you approach learning a new technology?",
    "Describe your ideal work environment.",
    "How do you handle feedback and criticism?",
    "What do you do when you are stuck on a problem?",
    "How do you manage your time?",
    "How do you handle disagreements with your manager?",
    "Are you comfortable working in an agile environment?",
    "How do you approach code reviews?",
    "Do you prefer frontend, backend, or full-stack work?",
    "How do you stay organized on large projects?",
    "How do you handle ambiguous requirements?",
    # ── Salary / Logistics ─────────────────────────────────────────────────────
    "What are your salary expectations?",
    "When can you start?",
    "Are you open to relocation?",
    "Do you have any questions for us?",
    "What are you looking for in your next position?",
    "Are you interviewing with other companies?",
    "What would make you say yes to this offer?",
]

# ---------------------------------------------------------------------------
# Class 2 — Noise (~110 examples)
# Key: includes CANDIDATE's own spoken answers (not questions), filler speech,
# background chatter, and off-topic statements.
# These are the most important negative examples — they prevent the system
# from processing what the CANDIDATE says in their own answer.
# ---------------------------------------------------------------------------
NOISE = [
    # ── Candidate's mid-answer fragments (most important to filter) ────────────
    "So I would start by initializing a pointer to null.",
    "The time complexity of my approach would be O of n log n.",
    "Let me trace through this with an example.",
    "So basically what happens is the left subtree stores smaller values.",
    "I think the answer here is related to memoization.",
    "The way I see it, you would use a hash map for constant lookup.",
    "Let me walk you through my thought process.",
    "I would approach this by first breaking it down into subproblems.",
    "So the reason we use a queue here is for FIFO ordering.",
    "In my experience working with React, hooks made this much cleaner.",
    "The tradeoff I see here is between time and space complexity.",
    "One thing I should mention is that this only works for sorted arrays.",
    "Does that answer your question?",
    "So to summarize what I just said.",
    "I think there is also an edge case here with empty input.",
    "Let me write out the pseudocode for this.",
    "The base case would be when the array is empty.",
    "I built something similar for a class project last semester.",
    "We used that at my last internship on the backend service.",
    "Actually, I think I need to reconsider that approach.",
    "Let me think about the edge cases here.",
    "So if I'm not mistaken, the worst case here is O of n squared.",
    "The key insight is that we can sort first and then binary search.",
    "I would probably use a hash set to track visited nodes.",
    "The recursive call would look something like this.",
    "So we maintain two pointers, one fast and one slow.",
    "I would start the timer and measure performance.",
    "In terms of implementation, I would use Python here.",
    "The helper function would take the root and return the height.",
    "We can avoid repeated work by caching results in a dictionary.",
    # ── Filler words / sounds ─────────────────────────────────────────────────
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
    "Right.",
    "Makes sense.",
    "Got it.",
    "Of course.",
    "Absolutely.",
    "Sure.",
    "Noted.",
    "Exactly.",
    "I agree.",
    "Okay I understand.",
    # ── Tech setup / interview logistics ─────────────────────────────────────
    "Can you hear me okay?",
    "Is the mic working?",
    "Let me adjust my camera.",
    "Can you see my screen?",
    "The internet connection seems slow.",
    "I think there is a lag.",
    "Let me share my screen.",
    "Sorry about that, my connection dropped.",
    "Can you see my code editor?",
    "Let me pull up the code.",
    "Give me a moment to open my IDE.",
    "Should I use an online editor?",
    "Let me close these other tabs.",
    # ── Random / off-topic ────────────────────────────────────────────────────
    "What is the capital of France?",
    "Did you see the football match last night?",
    "What did you have for lunch today?",
    "Can you pass me the coffee?",
    "I need to check my phone for a second.",
    "Let me get some water.",
    "Hold on I'll be right back.",
    "I am going to grab some coffee.",
    "I need a minute.",
    "Just a moment.",
    "Give me a second.",
    "I will be right back.",
    "Can you repeat that?",
    "Sorry I didn't catch that.",
    "Please go ahead.",
    "All right.",
    "What is the boiling point of water?",
    "Who won the World Cup?",
    "What year did World War 2 end?",
    # ── Short statements that sound like answers ──────────────────────────────
    "Yeah definitely.",
    "That reminds me of something I read.",
    "I have experience with that from university.",
    "Okay sounds good to me.",
    "Let me get started on that.",
    "I am not entirely sure about this one.",
    "I would need to look that up.",
    "That is a good point.",
    "I have not worked with that specifically.",
    "Let me think about that.",
    "I prefer to use Python for this.",
    "We used PostgreSQL in that project.",
    "It depends on the use case.",
    "There are a few ways to approach this.",
    "This is actually quite an interesting problem.",
    "I remember reading about this.",
    "I have not done that before but I am willing to learn.",
]

# ── Dynamic Loading from data.csv ──────────────────────────────────────────
csv_path = Path(__file__).parent.parent / "data.csv"
if csv_path.exists():
    import csv
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            added_tech = 0
            added_behav = 0
            
            # Simple heuristic keywords to classify behavioral vs technical
            behav_keywords = ["strength", "weakness", "tell me about", "describe a time", 
                              "why do you want", "how do you handle", "experience", "describe a"]
            
            for row in reader:
                q = row.get("output", "").strip()
                if q:
                    lower_q = q.lower()
                    if any(k in lower_q for k in behav_keywords):
                        PERSONAL_BEHAVIORAL.append(q)
                        added_behav += 1
                    else:
                        TECHNICAL_QUESTIONS.append(q)
                        added_tech += 1
        print(f"[DATASET] Loaded {added_tech} technical and {added_behav} behavioral questions from data.csv")
    except Exception as e:
        print(f"[DATASET] Failed to load data.csv: {e}")

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
    Embeds all examples and returns (X_train, X_test, y_train, y_test, embedder_name).
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
    print(f"[DATASET] Train: {len(X_train)} | Test: {len(X_test)}")
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
