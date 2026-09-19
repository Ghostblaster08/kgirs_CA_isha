"""
Virtual Laboratory Experiment: Create and Manage a Graph Database
Developed for Virtual Lab CA / IIT Kharagpur Virtual Labs Style.

A comprehensive, modular laboratory partitioned into 4 core sections:
  1. Theory: Concepts, architecture, LPG model, Cypher reference, setup, procedure, and terminology.
  2. Simulation: Interactive Property Graph Sandbox with visual UI controls, embedded Cypher query engine,
                 real-time Plotly graph visualization, graph metrics, and experimental trial logger.
  3. Quiz: 12-question self-grading conceptual assessment with immediate pedagogical explanations.
  4. Report Generation: Student information, recorded trials, graph snapshot, discussion, and downloadable PDF report.

Note: Streamlit-native components are strictly used to render seamlessly in both light and dark themes.
"""

import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional

warnings.filterwarnings("ignore", message=".*st.components.v1.html.*")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
import json
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Import embedded in-memory graph engine
from graph_engine import PropertyGraph, CypherEngine, Node, Relationship


# ======================================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "title": "Create and Manage a Graph Database",
    "course": "Database Management Systems / Advanced Data Systems",
    "lab_code": "CS-VLAB-08",
    "objectives": [
        "Understand the foundational principles of Graph Databases and the Labeled Property Graph (LPG) model.",
        "Differentiate Graph Databases from Relational Databases (RDBMS) via Index-Free Adjacency (IFA).",
        "Design, build, and manipulate graph structures consisting of nodes, labels, properties, and directed relationships.",
        "Master the Cypher Query Language for pattern matching, creation, updates, and deletions (MATCH, CREATE, SET, DELETE, DETACH DELETE).",
        "Execute single-hop and multi-hop relationship traversals, conditional filtering (WHERE), and aggregations (count, avg).",
        "Understand real-world graph database architecture, installation procedures (Neo4j Desktop, Docker, AuraDB), and production use cases."
    ]
}

THEORY_CONTENT = {
    "aim": (
        "To install and configure Neo4j (or an embedded Cypher-compatible engine), model connected domain entities "
        "using the Labeled Property Graph (LPG) paradigm, perform CRUD operations using the Cypher query language, "
        "and analyze relationship traversals across interconnected datasets."
    ),
    "learning_objectives": EXPERIMENT_CONFIG["objectives"],
    "introduction": """
### 1. Introduction to Graph Databases
A **Graph Database** is a purpose-built NoSQL database management system designed to store, manage, and query 
highly connected and complex relationship data. Unlike traditional relational database management systems (RDBMS) 
that organize data into rigid tabular rows and columns connected via foreign keys, graph databases treat 
**relationships as first-class citizens** stored directly alongside the entities they connect.

Modern applications—ranging from social media friend graphs, enterprise knowledge graphs, biomedical interaction 
networks, supply-chain logistics, to financial fraud rings—are characterized by dense, deep, and rapidly evolving 
connections. In these domains, traversing relationships in an RDBMS requires expensive, multi-way `JOIN` operations 
that experience exponential performance degradation ($O(N^k)$). Graph databases solve this challenge by adopting 
the **Property Graph Model** combined with **Index-Free Adjacency (IFA)**, enabling constant-time traversal ($O(1)$) 
per hop regardless of the total volume of data stored in the database.
    """,
    "rdbms_vs_graph": r"""
### 2. Graph Database vs. Relational Database (RDBMS)

| Feature / Dimension | Relational Database (RDBMS) | Graph Database (Neo4j / Property Graph) |
| :--- | :--- | :--- |
| **Primary Data Model** | Tables, Rows (tuples), Columns | Nodes (entities), Directed Relationships (edges) |
| **Relationship Storage** | Foreign keys & associative join tables | Direct physical memory pointers (Index-Free Adjacency) |
| **Deep Join Performance** | Degrades exponentially ($O(N^k)$) with join depth | Constant per-hop traversal time ($O(k)$), independent of total DB size |
| **Schema Flexibility** | Rigid DDL schema; costly schema migrations | Flexible schema / schema-optional; easily evolves |
| **Query Language** | SQL (Structured Query Language) | Declarative Graph Query Languages (Cypher, GQL) |
| **Multi-Hop Traversal** | Recursive Common Table Expressions (CTEs) | Intuitive ASCII-art pattern matching `(a)-[:REL*1..3]->(b)` |
| **Best-Fit Workloads** | Tabular transactions, accounting, structured reporting | Social networks, fraud detection, recommendation engines, knowledge graphs |

#### What is Index-Free Adjacency (IFA)?
In a relational database, traversing a relationship between two tables requires looking up foreign keys using an index 
(typically a $B^+$-Tree with $O(\log N)$ search complexity) or performing hash joins. When traversing multiple hops 
(e.g., *Find friends of friends of Alice*), each hop executes an independent index lookup across millions of records.

In **Neo4j and Native Graph Databases**, each node acts as a direct micro-index to its neighboring nodes. 
A node record holds direct 64-bit physical memory/disk offsets pointing to its connected relationship records, which in 
turn point directly to adjacent nodes. Therefore, traversing an edge requires only dereferencing a memory pointer ($O(1)$). 
The total query execution time is strictly proportional to the **size of the traversed subgraph**, completely independent 
of whether the entire database contains thousands or billions of nodes!
    """,
    "neo4j_architecture": """
### 3. Neo4j Architecture & Basic Concepts
Neo4j is the world's leading open-source native property graph database. Its core architecture consists of:

1. **Storage Layer (Native Graph Storage)**:
   - Neo4j stores graph structures in specialized, fixed-size record files:
     - `neostore.nodestore.db`: Fixed-length records (15 bytes) storing node in-use flags, pointer to the first relationship, and pointer to the first property.
     - `neostore.relationshipstore.db`: Fixed-length records (34 bytes) containing pointers to source node, target node, relationship type, previous and next relationships for both source and target nodes (doubly-linked relationship chain).
     - `neostore.propertystore.db`: Stores primitive properties (strings, integers, floats, booleans, arrays).
   - Because records are fixed-size, calculating a record's physical file offset is a simple multiplication: $\\text{Offset} = \\text{Record ID} \\times \\text{Record Size}$, enabling instant $O(1)$ random disk access.

2. **Execution Engine (Cypher Runtime)**:
   - Cypher queries are parsed into Abstract Syntax Trees (AST), validated, optimized by a cost-based query planner, and compiled into an executable pipeline.
   - Neo4j utilizes Volcano-style iterator models and pipelined batch runtimes to stream results with minimal memory overhead.

3. **Core Property Graph Elements**:
   - **Nodes**: Discrete domain entities (e.g., a student `Alice`, a course `DBMS`, a department `CSE`). Nodes can possess zero, one, or multiple labels.
   - **Labels**: Tags used to categorize nodes into semantic groups or roles (e.g., `:Student`, `:Faculty`, `:Course`). Labels serve as entry-point indexes.
   - **Properties**: Arbitrary key-value pairs associated with either nodes or relationships (e.g., `{name: 'Alice', gpa: 9.15}`).
   - **Relationships**: Directed connections between two nodes. Every relationship **must have a name/type** (e.g., `[:ENROLLED_IN]`), a designated start node, and an end node. Relationships can also hold properties (e.g., `{semester: '5th', grade: 'Ex'}`).
   - **Directionality**: While relationships are stored with a definite direction (from start node to end node), Cypher queries can traverse them in outgoing `(a)-[r]->(b)`, incoming `(a)<-[r]-(b)`, or bidirectional / undirected `(a)-[r]-(b)` modes.
    """,
    "cypher_crud": """
### 4. Cypher Query Language & CRUD Operations
Cypher is a declarative graph query language that utilizes visual, **ASCII-art syntax** to represent graph patterns.

#### Basic Pattern Grammar:
- **Nodes** are surrounded by parentheses: `(n)`, `(s:Student)`, `(s:Student {name: 'Alice'})`
- **Relationships** are enclosed in brackets with arrows: `-[r:ENROLLED_IN]->`, `<-[:TEACHES]-`, `-[r]-`
- **Paths** combine nodes and relationships: `(s:Student)-[:ENROLLED_IN]->(c:Course)`

#### Cypher CRUD Commands:
1. **CREATE (Create entities and connections)**:
   ```cypher
   // Create a new Student node
   CREATE (s:Student {id: 's_rahul', name: 'Rahul Sen', dept: 'CSE', gpa: 9.20})
   RETURN s;

   // Create a relationship between existing nodes
   MATCH (s:Student {id: 's_rahul'}), (c:Course {code: 'CS101'})
   CREATE (s)-[r:ENROLLED_IN {semester: '1st', grade: 'A'}]->(c)
   RETURN s, r, c;
   ```

2. **MATCH & RETURN (Read and pattern search)**:
   ```cypher
   // Find all courses a student is enrolled in
   MATCH (s:Student {name: 'Alice Smith'})-[r:ENROLLED_IN]->(c:Course)
   RETURN s.name, c.name, r.grade;
   ```

3. **WHERE (Filtering)**:
   ```cypher
   // Filter by numeric threshold and string pattern
   MATCH (s:Student)-[:ENROLLED_IN]->(c:Course)
   WHERE s.gpa >= 8.5 AND c.credits >= 4
   RETURN s.name, s.gpa, c.name, c.credits;
   ```

4. **SET & REMOVE (Update properties and labels)**:
   ```cypher
   // Update property value
   MATCH (s:Student {id: 's_rahul'})
   SET s.gpa = 9.40, s.status = 'Dean List'
   RETURN s;
   ```

5. **DELETE vs. DETACH DELETE (Delete nodes and relationships)**:
   - `DELETE n`: Deletes node `n`. **Constraint:** If node `n` has any attached relationships, Neo4j raises a `ConstraintViolationException` to preserve graph referential integrity.
   - `DETACH DELETE n`: Automatically deletes all incoming and outgoing relationships connected to `n`, then deletes the node itself.
   ```cypher
   // Delete relationship only
   MATCH (s:Student {id: 's_rahul'})-[r:ENROLLED_IN]->(c:Course {code: 'CS101'})
   DELETE r;

   // Safe deletion of node with relationships
   MATCH (s:Student {id: 's_rahul'})
   DETACH DELETE s;
   ```

6. **Aggregation & Grouping**:
   ```cypher
   // Count enrolled students per course (implicit GROUP BY)
   MATCH (c:Course)<-[:ENROLLED_IN]-(s:Student)
   RETURN c.name, count(s) AS total_students, avg(s.gpa) AS avg_gpa;
   ```

7. **Multi-Hop Traversal (Path Finding)**:
   ```cypher
   // Two-hop pattern: Find faculty who teach courses attended by Alice
   MATCH (s:Student {name: 'Alice Smith'})-[:ENROLLED_IN]->(c:Course)<-[:TEACHES]-(f:Faculty)
   RETURN s.name, c.name, f.name;

   // Variable-length prerequisite chain (1 to 3 hops)
   MATCH (c1:Course)-[:PREREQUISITE_OF*1..3]->(c2:Course)
   RETURN c1.name, c2.name;
   ```
    """,
    "setup_procedure": """
### 5. Prerequisites and Neo4j Installation Procedure

#### Prerequisites:
- **Java Virtual Machine (JVM)**: Java 17 LTS or Java 21 LTS (OpenJDK, Eclipse Temurin, or Oracle JDK).
- **Hardware**: Minimum 4 GB RAM (8 GB+ recommended), 64-bit OS (Windows, Linux, or macOS).

#### Deployment Options:

* **Option A: Neo4j Desktop (Recommended for GUI Learners)**
  1. Download **Neo4j Desktop** from [https://neo4j.com/download/](https://neo4j.com/download/).
  2. Install and launch the application. Create a new Project and click **Add -> Local DBMS**.
  3. Set a secure password for the default `neo4j` user and select Neo4j version `5.x`.
  4. Click **Start** to run the database server.
  5. Click **Open** to launch the integrated **Neo4j Browser** at `http://localhost:7474`.

* **Option B: Docker Container (Recommended for Developers & Labs)**
  Execute the following command in PowerShell or Terminal:
  ```powershell
  docker run -d `
    --name neo4j-vlab `
    -p 7474:7474 -p 7687:7687 `
    -e NEO4J_AUTH=neo4j/SecretPassword123 `
    -v neo4j_data:/data `
    neo4j:5.18.0
  ```
  - Port `7474`: HTTP web interface (Neo4j Browser).
  - Port `7687`: Bolt binary protocol for driver connections (Python, Java, Node.js).

* **Option C: Neo4j AuraDB (Free Cloud Instance)**
  1. Navigate to [https://neo4j.com/cloud/aura/](https://neo4j.com/cloud/aura/) and register for a free AuraDB Free instance.
  2. Save your generated database credentials (`neo4j+s://...`).
  3. Connect directly via the cloud-hosted Neo4j Workspace in your browser.

* **Option D: Cypher Shell (Command-Line Interface)**
  Launch the CLI client to execute queries interactively or run `.cypher` batch scripts:
  ```bash
  cypher-shell -u neo4j -p SecretPassword123
  ```
    """,
    "procedure": [
        "Step 1: Review the theoretical framework, LPG model, and Cypher syntax conventions.",
        "Step 2: Navigate to the 'Simulation' section from the left navigation menu.",
        "Step 3: Select and load a domain preset graph (e.g., University Academic Graph) or start with a blank graph.",
        "Step 4: Use the Visual Graph Builder tabs to create at least one new node (e.g., Student or Course) and one directed relationship.",
        "Step 5: Switch to the Cypher Query Editor tab. Load and execute pre-built example queries (MATCH, WHERE, aggregations).",
        "Step 6: Write custom Cypher queries to perform property updates (SET) and deletion (DETACH DELETE).",
        "Step 7: Inspect the real-time Plotly graph visualization and observe matched node highlighting.",
        "Step 8: Click 'Record Current State / Action' in the Data Log Book to log experimental trials.",
        "Step 9: Complete the 12-question Concept Assessment Quiz to test your graph database mastery.",
        "Step 10: Open 'Report Generation', enter your student credentials and analytical observations, and export your official PDF lab report."
    ],
    "precautions": """
### 6. Precautions & Common Mistakes in Graph Databases

1. **Attempting DELETE on Connected Nodes**:
   - Running `DELETE n` on a node with existing relationships triggers a referential constraint violation. Always use `DETACH DELETE n` if you intend to remove the node along with its connections.
2. **Unintended Cartesian Products in MATCH**:
   - Writing disconnected MATCH patterns like `MATCH (a:Student), (b:Course) RETURN a, b` computes an exhaustive Cartesian product of every student paired with every course ($O(V_1 \\times V_2)$). Always specify relationship patterns between entities: `MATCH (a)-[:ENROLLED_IN]->(b)`.
3. **Case Sensitivity in Cypher**:
   - Cypher keywords (`MATCH`, `WHERE`, `RETURN`) are case-insensitive, but **node labels** (`:Student`), **relationship types** (`[:ENROLLED_IN]`), and **property keys** (`name`, `gpa`) are strictly case-sensitive!
4. **Neglecting Relationship Directionality**:
   - Queries with directed arrows `(a)-[:REL]->(b)` will only return paths matching that exact traversal direction. Use undirected patterns `(a)-[:REL]-(b)` when bidirectional traversals are desired.
5. **Over-Indexing vs. Traversal**:
   - In graph databases, indexes should be created primarily on lookup properties (like student ID or email) to find initial starting nodes. Do not index relationships; graph traversal pointers (IFA) already provide instant traversal.
    """,
    "key_terms": {
        "Node (Vertex)": "A fundamental graph entity representing a distinct object (e.g., Student, Faculty, Course).",
        "Label": "A semantic tag applied to nodes for categorization, schema definition, and indexing (e.g., :Student, :Faculty, :Course).",
        "Relationship (Edge)": "A directed connection between two nodes with a mandatory type and direction (e.g., [:ENROLLED_IN]).",
        "Property": "A key-value attribute associated with a node or relationship (e.g., gpa: 9.15, credits: 4).",
        "Index-Free Adjacency (IFA)": "Architecture where nodes hold direct physical memory pointers to adjacent relationships and nodes, ensuring O(1) traversal.",
        "Cypher": "The declarative, ASCII-art pattern matching query language used by Neo4j and standardized under openCypher / GQL.",
        "DETACH DELETE": "A Cypher operation that safely deletes a node by first stripping all connected incoming and outgoing relationships.",
        "Degree of a Node": "The total number of relationships connected to a node (Degree = In-Degree + Out-Degree).",
        "Graph Density": "Ratio of existing relationships to the maximum possible directed relationships between nodes: D = E / (V * (V - 1)).",
        "Path": "A continuous alternating sequence of nodes and relationships connecting a start node to an end node.",
        "Multi-Hop Traversal": "A query path that navigates across two or more consecutive relationships (e.g., (a)-[]->(b)-[]->(c)).",
        "Isolated Node": "A node having a degree of zero (no incoming and no outgoing relationships)."
    }
}

# ======================================================================================
# 2. CONCEPT ASSESSMENT QUIZ (12 RIGOROUS QUESTIONS)
# ======================================================================================

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "What core architectural feature enables graph databases to achieve constant-time O(1) traversal per hop, unlike RDBMS multi-table joins?",
        "options": [
            "A) Distributed B-Tree indexes on foreign keys",
            "B) Index-Free Adjacency (IFA) using direct physical memory pointers",
            "C) Precomputed materialized relational views",
            "D) Columnar compressed storage files"
        ],
        "answer_index": 1,
        "explanation": "Index-Free Adjacency (IFA) means every node stores direct physical memory pointers to its adjacent relationships, allowing traversal in O(1) time without index searches."
    },
    {
        "id": 2,
        "question": "In the Neo4j Labeled Property Graph (LPG) model, which of the following statements regarding relationships is FALSE?",
        "options": [
            "A) Every relationship must have a start node and an end node",
            "B) Every relationship must have a specific type (e.g., [:ENROLLED_IN])",
            "C) Relationships can hold key-value properties just like nodes",
            "D) Relationships can exist as dangling pointers without a target node"
        ],
        "answer_index": 3,
        "explanation": "Relationships in Neo4j are strictly first-class directed connections. They can never exist as dangling pointers without both a valid source and target node."
    },
    {
        "id": 3,
        "question": "Which Cypher pattern correctly matches a Student named 'Alice' who is enrolled in any Course?",
        "options": [
            "A) SELECT Student WHERE name='Alice' JOIN Course",
            "B) MATCH (s:Student {name: 'Alice'})-[:ENROLLED_IN]->(c:Course) RETURN s, c",
            "C) FIND (s:Student)-[ENROLLED_IN]->(c:Course) FILTER s.name = 'Alice'",
            "D) MATCH {s:Student} --> {c:Course} WHERE name = 'Alice'"
        ],
        "answer_index": 1,
        "explanation": "Cypher uses ASCII-art syntax: nodes are enclosed in parentheses '(s:Student)' and directed relationships in bracketed arrows '-[:ENROLLED_IN]->'."
    },
    {
        "id": 4,
        "question": "What happens if you execute 'MATCH (n:Student {id: 'S01'}) DELETE n' when node 'S01' currently has 3 active relationships?",
        "options": [
            "A) The node and its 3 relationships are automatically deleted without error",
            "B) The 3 relationships are preserved as dangling pointers with null sources",
            "C) Neo4j throws a ConstraintViolationException preventing deletion to preserve referential integrity",
            "D) The node is deleted and the target nodes are also recursively deleted"
        ],
        "answer_index": 2,
        "explanation": "To preserve graph referential integrity, plain DELETE fails if relationships are attached. 'DETACH DELETE' must be explicitly used to remove attached relationships first."
    },
    {
        "id": 5,
        "question": "What is the primary operational role of 'Labels' attached to nodes in Neo4j?",
        "options": [
            "A) To store arbitrary floating-point numeric measurements",
            "B) To categorize nodes into domain groups and act as entry-point indexes for fast query lookup",
            "C) To define foreign key cascade rules between tables",
            "D) Labels are purely cosmetic and have no execution impact"
        ],
        "answer_index": 1,
        "explanation": "Labels group nodes into semantic roles (e.g., :Student, :Faculty) and allow Neo4j to index and rapidly locate starting nodes for graph traversals."
    },
    {
        "id": 6,
        "question": "Which Cypher statement correctly updates the GPA of student 'Alice' to 9.50 and adds an 'honors' property?",
        "options": [
            "A) UPDATE (s:Student {name: 'Alice'}) SET gpa = 9.50, honors = true",
            "B) MATCH (s:Student {name: 'Alice'}) SET s.gpa = 9.50, s.honors = true RETURN s",
            "C) MODIFY Student Alice (gpa: 9.50, honors: true)",
            "D) ALTER NODE (s:Student) WHERE name='Alice' ADD gpa=9.50"
        ],
        "answer_index": 1,
        "explanation": "In Cypher, property updates are performed using the 'SET' clause following a 'MATCH' pattern: 'MATCH (s) SET s.prop = val'."
    },
    {
        "id": 7,
        "question": "In Cypher, what is the behavior of the aggregation function 'count(s)' in 'MATCH (c:Course)<-[:ENROLLED_IN]-(s:Student) RETURN c.name, count(s)'?",
        "options": [
            "A) It throws a syntax error because Cypher requires an explicit 'GROUP BY' clause",
            "B) It automatically groups by non-aggregated fields (c.name) and counts students per course",
            "C) It counts all students in the database regardless of course",
            "D) It only counts courses, ignoring students completely"
        ],
        "answer_index": 1,
        "explanation": "Unlike SQL, Cypher has implicit grouping: any non-aggregated expressions in the RETURN clause (such as c.name) automatically serve as grouping keys."
    },
    {
        "id": 8,
        "question": "What does the variable-length Cypher relationship pattern '-[:PREREQUISITE_OF*1..3]->' express?",
        "options": [
            "A) A relationship whose weight is between 1.0 and 3.0",
            "B) A path of between 1 and 3 sequential PREREQUISITE_OF hops between entities",
            "C) A relationship that must be traversed exactly 3 times in a loop",
            "D) An array of 3 distinct relationship property keys"
        ],
        "answer_index": 1,
        "explanation": "Syntax '*minHops..maxHops' specifies variable-length path traversal. '*1..3' searches for paths having from 1 up to 3 consecutive relationship hops."
    },
    {
        "id": 9,
        "question": "In which scenario would a Relational Database (RDBMS) typically outperform a Graph Database?",
        "options": [
            "A) Finding mutual friends across 6 degrees of separation in a social network",
            "B) Detecting circular money laundering rings across transaction accounts",
            "C) Sequential bulk aggregations across millions of independent, flat accounting records",
            "D) Finding shortest paths through an international airline flight network"
        ],
        "answer_index": 2,
        "explanation": "Relational databases and columnar engines excel at bulk, linear scans and aggregations across flat tables with minimal inter-record joins, whereas graph databases excel at complex, multi-hop relationship traversals."
    },
    {
        "id": 10,
        "question": "Why is the query 'MATCH (s:Student), (c:Course) RETURN s, c' generally discouraged unless explicitly intended?",
        "options": [
            "A) Because it causes a syntax error in Cypher",
            "B) Because it produces an unconstrained Cartesian product matching every student with every course",
            "C) Because it automatically deletes all students and courses",
            "D) Because it forces the database to convert into an RDBMS table"
        ],
        "answer_index": 1,
        "explanation": "Matching disconnected entities without relationship patterns calculates a full Cartesian product (O(|V1| * |V2|)), which can consume massive memory on large graphs."
    },
    {
        "id": 11,
        "question": "Which of the following describes the default network port used for Bolt protocol driver communication in Neo4j?",
        "options": [
            "A) Port 7474 (HTTP Browser interface)",
            "B) Port 7687 (Bolt binary protocol)",
            "C) Port 3306 (MySQL default port)",
            "D) Port 5432 (PostgreSQL default port)"
        ],
        "answer_index": 1,
        "explanation": "Neo4j uses port 7474 for HTTP / Neo4j Browser access, and port 7687 for high-performance Bolt binary protocol connections utilized by official drivers."
    },
    {
        "id": 12,
        "question": "According to Cypher naming conventions and syntax standards, how should relationship types and node labels be cased?",
        "options": [
            "A) Labels in UPPER_CASE and Relationships in lower_case",
            "B) Labels in UpperCamelCase (e.g., :Student) and Relationships in UPPER_SNAKE_CASE (e.g., [:ENROLLED_IN])",
            "C) Both labels and relationships must always be lowercase",
            "D) Cypher forbids the use of underscores in relationship types"
        ],
        "answer_index": 1,
        "explanation": "Cypher naming conventions dictate UpperCamelCase for Node Labels (e.g. :Student, :Course) and UPPER_SNAKE_CASE for Relationship Types (e.g. [:ENROLLED_IN], [:TEACHES])."
    }
]


# ======================================================================================
# 3. LAB REPORT PDF EXPORTER
# ======================================================================================

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 130, 140)
        self.cell(0, 10, f"Page {self.page_no()} | Virtual Laboratory CA - Neo4j Graph Database Experiment", align="C")


def generate_pdf_report(student_name: str, student_id: str, date_str: str,
                        trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                        student_notes: str, graph_metrics: Dict[str, Any]) -> bytes:
    """Compiles experiment benchmark records into an official, publication-quality PDF report."""
    pdf = LabReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Document Header
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, EXPERIMENT_CONFIG["title"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 6, f"{EXPERIMENT_CONFIG['course']} | Course Code: {EXPERIMENT_CONFIG['lab_code']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # Student & Session Info Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 27, 190, 22, "FD")

    pdf.set_xy(14, 29)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student Name:", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, student_name or "N/A", new_x=XPos.RIGHT, new_y=YPos.TOP)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Roll / ID Number:", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, student_id or "N/A", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_xy(14, 38)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Experiment Date:", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, date_str or datetime.now().strftime("%Y-%m-%d"), new_x=XPos.RIGHT, new_y=YPos.TOP)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Evaluation:", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "B", 9)
    pct = int((quiz_score / quiz_total) * 100 if quiz_total else 0)
    if pct >= 50:
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(50, 5, f"{quiz_score} / {quiz_total} ({pct}%)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(12)

    # 1. Learning Objectives
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Experiment Objectives & Aim", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        clean_obj = str(obj).replace("$", "").replace("\\", "")
        pdf.cell(5, 4.5, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(0, 4.5, f" {clean_obj}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # 2. Final Graph Snapshot & Metrics
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Final Graph State & Metric Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)

    summary_text = (
        f"Total Nodes: {graph_metrics.get('num_nodes', 0)}   |   "
        f"Total Relationships: {graph_metrics.get('num_relationships', 0)}   |   "
        f"Labels: {', '.join(graph_metrics.get('labels', [])) or 'None'}\n"
        f"Rel Types: {', '.join(graph_metrics.get('rel_types', [])) or 'None'}   |   "
        f"Graph Density: {graph_metrics.get('density', 0.0)}   |   "
        f"Average Degree: {graph_metrics.get('avg_degree', 0.0)}"
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(4)

    # 3. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Recorded Experimental Trials & Execution Log", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No experimental simulation trials recorded during this session.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)

        # Fixed column widths matching 190 mm
        col_widths = {
            "Trial #": 14,
            "Operation": 28,
            "Query / Action": 68,
            "Result": 42,
            "Status": 18,
            "Timestamp": 20
        }

        # Header
        for c in trials_df.columns:
            w = col_widths.get(c, 25)
            pdf.cell(w, 6, str(c)[:16], border=1, align="C", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()

        # Rows
        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 7.5)
        fill = False

        for _, row in trials_df.iterrows():
            for c in trials_df.columns:
                w = col_widths.get(c, 25)
                val_str = str(row[c])
                pdf.cell(w, 5, val_str[:38], border=1, align="C", fill=fill, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.ln()
            fill = not fill

    pdf.ln(5)

    # 4. Student Discussion & Observations
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "4. Student Observations & Analytical Discussion", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "The graph database experiment successfully demonstrated node and relationship creation, "
        "Cypher pattern matching, property updates, and referential constraints (DETACH DELETE). "
        "Graph traversal queries executed with high efficiency without requiring relational table joins."
    )
    pdf.multi_cell(0, 5, notes_text)
    pdf.ln(8)

    # Sign-off line
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Student Signature", align="C")

    return bytes(pdf.output())


# ======================================================================================
# 4. GRAPH PRESETS, SCHEMAS & VISUALIZERS
# ======================================================================================

PRESET_SCHEMAS = {
    "University Academic Knowledge Graph (Default)": {
        "node_labels": ["Student", "Course", "Faculty", "Department", "Project", "Custom..."],
        "default_properties": {
            "Student": {"id_prefix": "s_", "name": "Kiran Patel", "key1": "dept", "val1": "CSE", "key2": "gpa", "val2": "8.80"},
            "Course": {"id_prefix": "cs_", "name": "Machine Learning", "key1": "code", "val1": "CS401", "key2": "credits", "val2": "4"},
            "Faculty": {"id_prefix": "prof_", "name": "Prof. V. Rao", "key1": "dept", "val1": "CSE", "key2": "role", "val2": "Professor"},
            "Department": {"id_prefix": "dept_", "name": "Information Technology", "key1": "code", "val1": "IT", "key2": "building", "val2": "Aryabhatta"},
            "Project": {"id_prefix": "proj_", "name": "Graph Neural Networks", "key1": "domain", "val1": "AI", "key2": "budget", "val2": "150000"},
            "Custom...": {"id_prefix": "custom_", "name": "Custom Entity", "key1": "type", "val1": "General", "key2": "status", "val2": "Active"},
        },
        "rel_types": ["ENROLLED_IN", "TEACHES", "PREREQUISITE_OF", "BELONGS_TO", "ADVISES", "OFFERED_BY", "Custom..."],
        "default_rel_properties": {
            "ENROLLED_IN": {"key": "grade", "val": "A"},
            "TEACHES": {"key": "academic_year", "val": "2024-25"},
            "PREREQUISITE_OF": {"key": "mandatory", "val": "True"},
            "BELONGS_TO": {"key": "since", "val": "2018"},
            "ADVISES": {"key": "project", "val": "Graph Optimization"},
            "OFFERED_BY": {"key": "semester", "val": "Autumn"},
            "Custom...": {"key": "weight", "val": "1.0"},
        },
        "presentation_example": {
            "title": "Add ResearchLab Entity (University Academic Graph)",
            "scenario": "Demonstrate adding a new Research Laboratory entity and linking a Faculty member as director.",
            "label": "ResearchLab",
            "id": "lab_nlp",
            "name": "NLP & AI Research Lab",
            "key1": "director",
            "val1": "Prof. S. Banerjee",
            "key2": "grants_inr",
            "val2": "5000000",
            "followup": "Connect prof_banerjee -[:DIRECTS]-> lab_nlp"
        },
        "cypher_examples": {
            "-- Select an Educational Cypher Example --": "",
            "1. Match all nodes and inspect entire graph": "MATCH (n) RETURN n",
            "2. Match enrolled students and their courses": "MATCH (s:Student)-[:ENROLLED_IN]->(c:Course) RETURN s.name, c.name, s.gpa",
            "3. Filter high-performing students (WHERE clause)": "MATCH (s:Student) WHERE s.gpa >= 8.5 RETURN s.name, s.dept, s.gpa",
            "4. Multi-hop traversal: Faculty teaching enrolled students": "MATCH (f:Faculty)-[:TEACHES]->(c:Course)<-[:ENROLLED_IN]-(s:Student) RETURN f.name, c.name, s.name",
            "5. Aggregate enrollment counts per course (count)": "MATCH (c:Course)<-[:ENROLLED_IN]-(s:Student) RETURN c.name, count(s) AS total_enrolled",
            "6. Prerequisite chain traversal (Course -> Course)": "MATCH (c1:Course)-[:PREREQUISITE_OF]->(c2:Course) RETURN c1.name, c2.name",
            "7. CREATE a new Student node": "CREATE (s:Student {id: 's_kiran', name: 'Kiran Patel', dept: 'CSE', gpa: 8.75}) RETURN s",
            "8. Connect new Student to Course (CREATE relationship)": "MATCH (s:Student {id: 's_kiran'}), (c:Course {name: 'Database Management Systems'}) CREATE (s)-[:ENROLLED_IN {grade: 'A', semester: '5th'}]->(c) RETURN s, c",
            "9. Update student GPA using SET": "MATCH (s:Student {id: 's_kiran'}) SET s.gpa = 9.40 RETURN s",
            "10. Safely remove student using DETACH DELETE": "MATCH (s:Student {id: 's_kiran'}) DETACH DELETE s"
        }
    },
    "Social Network & Friendships": {
        "node_labels": ["Person", "Group", "Topic", "Event", "Custom..."],
        "default_properties": {
            "Person": {"id_prefix": "p_", "name": "Elena Rostova", "key1": "city", "val1": "Mumbai", "key2": "age", "val2": "25"},
            "Group": {"id_prefix": "g_", "name": "Deep Learning Club", "key1": "members_count", "val1": "220", "key2": "category", "val2": "Technology"},
            "Topic": {"id_prefix": "i_", "name": "Knowledge Graphs", "key1": "domain", "val1": "Databases", "key2": "level", "val2": "Advanced"},
            "Event": {"id_prefix": "e_", "name": "Graph Summit 2024", "key1": "location", "val1": "Virtual", "key2": "attendees", "val2": "500"},
            "Custom...": {"id_prefix": "custom_", "name": "Custom Entity", "key1": "category", "val1": "Social", "key2": "active", "val2": "True"},
        },
        "rel_types": ["FRIENDS_WITH", "MEMBER_OF", "INTERESTED_IN", "FOLLOWS", "ORGANIZED", "Custom..."],
        "default_rel_properties": {
            "FRIENDS_WITH": {"key": "since", "val": "2023"},
            "MEMBER_OF": {"key": "role", "val": "Moderator"},
            "INTERESTED_IN": {"key": "level", "val": "Expert"},
            "FOLLOWS": {"key": "since", "val": "2024"},
            "ORGANIZED": {"key": "role", "val": "Lead Organizer"},
            "Custom...": {"key": "weight", "val": "1.0"},
        },
        "presentation_example": {
            "title": "Add CommunityLeader Entity (Social Graph)",
            "scenario": "Demonstrate adding a new influencer or community leader and linking them to existing social groups.",
            "label": "Person",
            "id": "p_elena",
            "name": "Elena Rostova",
            "key1": "city",
            "val1": "Mumbai",
            "key2": "followers",
            "val2": "12500",
            "followup": "Connect p_elena -[:MEMBER_OF]-> g_ai"
        },
        "cypher_examples": {
            "-- Select an Educational Cypher Example --": "",
            "1. Match all people and their cities": "MATCH (p:Person) RETURN p.name, p.city, p.age",
            "2. Find friendships in network": "MATCH (p1:Person)-[:FRIENDS_WITH]->(p2:Person) RETURN p1.name, p2.name",
            "3. Group memberships by person": "MATCH (p:Person)-[m:MEMBER_OF]->(g:Group) RETURN p.name, g.name, m.role",
            "4. Shared topics of interest": "MATCH (p:Person)-[:INTERESTED_IN]->(t:Topic) RETURN p.name, t.name, t.domain",
            "5. Friend-of-friend multi-hop traversal": "MATCH (p1:Person)-[:FRIENDS_WITH]->(p2:Person)-[:FRIENDS_WITH]->(p3:Person) RETURN p1.name, p3.name",
            "6. Connect friends using CREATE": "MATCH (p1:Person {name: 'Alex'}), (p2:Person {name: 'Chris'}) CREATE (p1)-[:FRIENDS_WITH {since: 2024}]->(p2) RETURN p1, p2",
            "7. Count members in each group": "MATCH (p:Person)-[:MEMBER_OF]->(g:Group) RETURN g.name, count(p) AS total_members",
            "8. Delete relationship using DELETE": "MATCH (p:Person {name: 'Dan'})-[r:MEMBER_OF]->() DELETE r"
        }
    },
    "Financial Fraud Detection Ring": {
        "node_labels": ["Account", "Device", "IPAddress", "Merchant", "Custom..."],
        "default_properties": {
            "Account": {"id_prefix": "acc_", "name": "ACC-105", "key1": "owner", "val1": "David", "key2": "balance", "val2": "42000"},
            "Device": {"id_prefix": "dev_", "name": "DEV-MAC-9931", "key1": "device_id", "val1": "DEV-MAC-9931", "key2": "os", "val2": "iOS 17"},
            "IPAddress": {"id_prefix": "ip_", "name": "192.168.1.188", "key1": "ip", "val1": "192.168.1.188", "key2": "city", "val2": "Delhi"},
            "Merchant": {"id_prefix": "merch_", "name": "CryptoPay Gateway", "key1": "category", "val1": "Crypto", "key2": "risk_score", "val2": "95"},
            "Custom...": {"id_prefix": "custom_", "name": "Custom Entity", "key1": "risk_level", "val1": "HIGH", "key2": "status", "val2": "Flagged"},
        },
        "rel_types": ["USED_DEVICE", "LOGGED_FROM", "TRANSFER", "FLAGGED_WITH", "PAID_TO", "Custom..."],
        "default_rel_properties": {
            "USED_DEVICE": {"key": "last_login", "val": "2024-09-10"},
            "LOGGED_FROM": {"key": "timestamp", "val": "14:32:10"},
            "TRANSFER": {"key": "amount", "val": "8500"},
            "FLAGGED_WITH": {"key": "risk_score", "val": "88"},
            "PAID_TO": {"key": "amount", "val": "12000"},
            "Custom...": {"key": "weight", "val": "1.0"},
        },
        "presentation_example": {
            "title": "Add High-Risk Merchant Entity (Fraud Graph)",
            "scenario": "Demonstrate adding a high-risk crypto merchant or mule account to investigate laundering.",
            "label": "Merchant",
            "id": "merch_crypto",
            "name": "CryptoPay Ltd",
            "key1": "risk_level",
            "val1": "CRITICAL",
            "key2": "license",
            "val2": "Offshore-KYC-Bypass",
            "followup": "Connect acc_104 -[:TRANSFER {amount: 9500}]-> merch_crypto"
        },
        "cypher_examples": {
            "-- Select an Educational Cypher Example --": "",
            "1. Match all accounts and balances": "MATCH (a:Account) RETURN a.acc_no, a.owner, a.balance",
            "2. Detect circular money transfers (Fraud Ring)": "MATCH (a1:Account)-[t1:TRANSFER]->(a2:Account)-[t2:TRANSFER]->(a3:Account)-[t3:TRANSFER]->(a1) RETURN a1.owner, a2.owner, a3.owner, t1.amount, t2.amount, t3.amount",
            "3. Find shared devices across accounts": "MATCH (a:Account)-[:USED_DEVICE]->(d:Device) RETURN d.device_id, count(a) AS linked_accounts",
            "4. Trace accounts sharing same IP address": "MATCH (a:Account)-[:LOGGED_FROM]->(ip:IPAddress) RETURN ip.ip, ip.city, a.owner",
            "5. High-value transactions filter (WHERE amount >= 4500)": "MATCH (a1:Account)-[t:TRANSFER]->(a2:Account) WHERE t.amount >= 4500 RETURN a1.owner, a2.owner, t.amount",
            "6. Flag suspicious account with updated balance": "MATCH (a:Account {acc_no: 'ACC-104'}) SET a.balance = 0 RETURN a"
        }
    },
    "Blank / Empty Graph": {
        "node_labels": ["Entity", "User", "Concept", "Item", "Category", "Custom..."],
        "default_properties": {
            "Entity": {"id_prefix": "node_", "name": "Root Node", "key1": "type", "val1": "Base", "key2": "value", "val2": "100"},
            "User": {"id_prefix": "user_", "name": "Alex", "key1": "role", "val1": "Admin", "key2": "active", "val2": "True"},
            "Concept": {"id_prefix": "concept_", "name": "Graph Theory", "key1": "domain", "val1": "Mathematics", "key2": "difficulty", "val2": "Introductory"},
            "Item": {"id_prefix": "item_", "name": "Item A", "key1": "sku", "val1": "SKU001", "key2": "price", "val2": "49.99"},
            "Category": {"id_prefix": "cat_", "name": "Electronics", "key1": "code", "val1": "ELEC", "key2": "priority", "val2": "High"},
            "Custom...": {"id_prefix": "custom_", "name": "Custom Entity", "key1": "property_key", "val1": "property_val", "key2": "status", "val2": "Active"},
        },
        "rel_types": ["CONNECTED_TO", "RELATES_TO", "PART_OF", "DEPENDS_ON", "LINKED_WITH", "Custom..."],
        "default_rel_properties": {
            "CONNECTED_TO": {"key": "weight", "val": "1.0"},
            "RELATES_TO": {"key": "context", "val": "General"},
            "PART_OF": {"key": "order", "val": "1"},
            "DEPENDS_ON": {"key": "required", "val": "True"},
            "LINKED_WITH": {"key": "since", "val": "2024"},
            "Custom...": {"key": "weight", "val": "1.0"},
        },
        "presentation_example": {
            "title": "Add Custom Node from Scratch (Blank Graph)",
            "scenario": "Demonstrate creating custom domain entities and connecting them with directed relationships.",
            "label": "City",
            "id": "city_kgp",
            "name": "Kharagpur",
            "key1": "state",
            "val1": "West Bengal",
            "key2": "pin_code",
            "val2": "721302",
            "followup": "Add second node 'city_kolkata' and connect with CONNECTED_TO"
        },
        "cypher_examples": {
            "-- Select an Educational Cypher Example --": "",
            "1. Match all nodes in the graph": "MATCH (n) RETURN n",
            "2. Create custom node with properties": "CREATE (n:CustomEntity {id: 'c1', name: 'Sample Entity', priority: 'High'}) RETURN n",
            "3. Connect nodes with a relationship": "MATCH (a), (b) WHERE a.id = 'c1' AND b.id = 'c2' CREATE (a)-[:CONNECTED_TO {weight: 1.5}]->(b) RETURN a, b",
            "4. Detach delete all nodes (Reset)": "MATCH (n) DETACH DELETE n"
        }
    }
}

LABEL_COLORS = {
    "Student": "#2563EB",       # Blue
    "Course": "#F59E0B",        # Amber
    "Faculty": "#10B981",       # Emerald Green
    "Department": "#8B5CF6",    # Purple
    "Project": "#3B82F6",       # Bright Blue
    "Person": "#06B6D4",        # Cyan
    "Group": "#EC4899",         # Pink
    "Topic": "#14B8A6",         # Teal
    "Event": "#F43F5E",         # Rose
    "Account": "#EF4444",       # Red
    "Device": "#64748B",        # Slate
    "IPAddress": "#F97316",     # Orange
    "Merchant": "#DC2626",      # Crimson Red
    "ResearchLab": "#8B5CF6",   # Violet
    "City": "#0EA5E9",          # Sky Blue
    "Entity": "#6366F1",        # Indigo
    "User": "#2563EB",          # Blue
    "Concept": "#10B981",       # Emerald
    "Item": "#F59E0B",          # Amber
    "Category": "#A855F7"       # Purple
}
DEFAULT_NODE_COLOR = "#6366F1"  # Indigo

def get_node_color(label: str) -> str:
    """Returns color for a node label with deterministic fallback."""
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    palette = ["#6366F1", "#EC4899", "#14B8A6", "#F59E0B", "#10B981", "#8B5CF6", "#06B6D4", "#F97316", "#3B82F6"]
    return palette[abs(hash(label)) % len(palette)]


def get_primary_label(labels: Any) -> str:
    """Returns the primary label for a node, prioritizing domain entities (e.g. Student, Faculty) over generic labels."""
    if not labels:
        return "Entity"
    lbl_list = list(labels)
    # Prefer specific domain entities like Student, Faculty in academic/university context
    priority = ["Student", "Faculty", "Course", "Department", "Project", "Account", "Device", "Merchant", "Group", "Topic"]
    for pref in priority:
        if pref in lbl_list:
            return pref
    return sorted(lbl_list)[0]


def render_graph_figure(graph: PropertyGraph,
                         matched_node_ids: Optional[List[str]] = None,
                         matched_rel_ids: Optional[List[str]] = None,
                         layout_algorithm: str = "Spring (Force-Directed)",
                         node_label_mode: str = "Name / Label",
                         theme: str = "auto") -> go.Figure:
    """
    Renders an interactive 2D graph visualization using NetworkX for layout
    and Plotly for interactive rendering with hovercards and arrows.
    """
    fig = go.Figure()

    if not graph.nodes:
        fig.add_annotation(
            text="Graph is currently empty. Add nodes or load a preset dataset!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#64748B")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    # Build NetworkX representation for coordinates
    G = graph.to_networkx()

    # Layout computation
    n_count = len(G.nodes())
    if layout_algorithm == "Circular":
        pos = nx.circular_layout(G)
    elif layout_algorithm == "Kamada-Kawai":
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=max(0.6, 2.5 / np.sqrt(max(1, n_count))))
    elif layout_algorithm == "Shell":
        label_groups = {}
        for nid, node in graph.nodes.items():
            primary_lbl = get_primary_label(node.labels)
            label_groups.setdefault(primary_lbl, []).append(nid)
        pos = nx.shell_layout(G, nlist=list(label_groups.values()))
    else:  # Default Spring
        pos = nx.spring_layout(G, seed=42, k=max(0.7, 3.0 / np.sqrt(max(1, n_count))), iterations=60)

    matched_nids_set = set(matched_node_ids) if matched_node_ids else set()
    matched_rids_set = set(matched_rel_ids) if matched_rel_ids else set()

    # 1. Edge Line Traces
    edge_x = []
    edge_y = []
    mid_x = []
    mid_y = []
    mid_text = []
    mid_color = []

    for rid, rel in graph.relationships.items():
        if rel.source not in pos or rel.target not in pos:
            continue
        x0, y0 = pos[rel.source]
        x1, y1 = pos[rel.target]

        # Line segment
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Directional point
        mx = x0 + 0.60 * (x1 - x0)
        my = y0 + 0.60 * (y1 - y0)
        mid_x.append(mx)
        mid_y.append(my)

        is_matched = rid in matched_rids_set
        color = "#F59E0B" if is_matched else "#64748B"
        mid_color.append(color)
        mid_text.append(f"<b>{rel.type}</b>")

    # Base Edge Lines
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=2, color="#94A3B8"),
        hoverinfo="none",
        showlegend=False
    ))

    # Relationship Labels & Direction Indicators
    if mid_x:
        fig.add_trace(go.Scatter(
            x=mid_x, y=mid_y,
            mode="text+markers",
            marker=dict(size=10, symbol="triangle-up", color=mid_color, line=dict(width=1, color="#64748B")),
            text=mid_text,
            textposition="top center",
            textfont=dict(size=10, color="#475569", family="Arial, sans-serif"),
            hoverinfo="none",
            showlegend=False
        ))

    # 2. Node Traces (Grouped by Primary Label for clean legend)
    nodes_by_label: Dict[str, List[str]] = {}
    for nid, node in graph.nodes.items():
        primary_label = get_primary_label(node.labels)
        nodes_by_label.setdefault(primary_label, []).append(nid)

    for label_name, nids in nodes_by_label.items():
        nx_coords = []
        ny_coords = []
        display_texts = []
        sizes = []
        line_widths = []
        line_colors = []

        base_color = get_node_color(label_name)

        for nid in nids:
            if nid not in pos:
                continue
            x, y = pos[nid]
            nx_coords.append(x)
            ny_coords.append(y)

            node = graph.nodes[nid]
            is_matched = nid in matched_nids_set

            if is_matched:
                sizes.append(45)
                line_widths.append(4)
                line_colors.append("#FDE047")  # Bright Yellow Highlight
            else:
                sizes.append(36)
                line_widths.append(2.5)
                line_colors.append("#FFFFFF")

            # Display text
            if node_label_mode == "Name / Label":
                display_texts.append(f"<b>{node.display_name()}</b><br><i>:{label_name}</i>")
            elif node_label_mode == "Name Only":
                display_texts.append(f"<b>{node.display_name()}</b>")
            elif node_label_mode == "Node ID":
                display_texts.append(f"<b>{nid}</b>")
            elif node_label_mode == "Label Only":
                display_texts.append(f"<b>:{label_name}</b>")
            else:
                display_texts.append("")

        fig.add_trace(go.Scatter(
            x=nx_coords, y=ny_coords,
            mode="markers+text",
            name=f":{label_name} ({len(nids)})",
            marker=dict(
                size=sizes,
                color=base_color,
                line=dict(width=line_widths, color=line_colors),
                opacity=0.95
            ),
            text=display_texts,
            textposition="bottom center",
            textfont=dict(size=12, color="#0F172A", family="Arial, sans-serif"),
            hoverinfo="none"
        ))

    # Matched highlight legend indicator if any
    if matched_node_ids:
        fig.add_annotation(
            text=f"✨ Matched Subgraph: {len(matched_node_ids)} Node(s) highlighted",
            xref="paper", yref="paper",
            x=0.01, y=0.99, showarrow=False,
            bgcolor="#FEF08A",
            font=dict(size=12, color="#854D0E", family="Arial, sans-serif"),
            bordercolor="#FACC15",
            borderwidth=2,
            borderpad=6,
            opacity=0.95
        )

    fig.update_layout(
        title=dict(text="<b>Graph Topology</b>", font=dict(size=18, family="Arial, sans-serif", color="#0F172A")),
        hovermode=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        margin=dict(l=15, r=15, t=50, b=20),
        plot_bgcolor="rgba(248,250,252,0.6)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#0F172A"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E2E8F0",
            borderwidth=1
        )
    )

    return fig


def render_interactive_graph_canvas(graph: PropertyGraph, 
                                    matched_node_ids: Optional[List[str]] = None,
                                    matched_rel_ids: Optional[List[str]] = None) -> None:
    """Renders a Neo4j Bloom style interactive graph using vis-network."""
    
    nodes_data = []
    edges_data = []
    
    matched_nids = set(matched_node_ids) if matched_node_ids else set()
    matched_rids = set(matched_rel_ids) if matched_rel_ids else set()
    
    for nid, node in graph.nodes.items():
        primary_label = get_primary_label(node.labels)
        base_color = get_node_color(primary_label)
        
        is_matched = nid in matched_nids
        border_width = 4 if is_matched else 2
        border_color = "#FACC15" if is_matched else "#FFFFFF"
        
        # Build hover title (HTML)
        labels_str = ":" + ":".join(sorted(node.labels))
        prop_lines = "".join([f"<tr><td style='padding-right:8px;'><b>{k}</b></td><td>{v}</td></tr>" for k,v in node.properties.items()])
        title_html = f"<div style='font-family: Arial, sans-serif; padding:5px;'><b style='color:#1E293B; font-size:14px;'>{node.display_name()}</b><br><span style='color:#6366F1; font-size:12px;'>{labels_str}</span><br><span style='color:#94A3B8; font-size:10px;'>ID: {nid}</span>"
        if prop_lines:
            title_html += f"<hr style='margin:4px 0;'><table style='font-size:11px; color:#334155;'>{prop_lines}</table>"
        title_html += "</div>"
        
        nodes_data.append({
            "id": nid,
            "label": f"<b>{node.display_name()}</b>\n<i>:{primary_label}</i>",
            "title": title_html,
            "color": {
                "background": base_color,
                "border": border_color,
                "highlight": {"background": base_color, "border": "#F59E0B"},
                "hover": {"background": base_color, "border": "#94A3B8"}
            },
            "borderWidth": border_width,
            "borderWidthSelected": 4,
            "shape": "dot",
            "size": 32,
            "font": {"size": 12, "color": "#0F172A", "face": "Arial, sans-serif", "multi": "html", "align": "center"}
        })
        
    for rid, rel in graph.relationships.items():
        is_matched = rid in matched_rids
        edge_color = "#F59E0B" if is_matched else "#94A3B8"
        
        prop_lines = "".join([f"<tr><td style='padding-right:8px;'><b>{k}</b></td><td>{v}</td></tr>" for k,v in rel.properties.items()])
        title_html = f"<div style='font-family: Arial, sans-serif; padding:5px;'><b style='color:#334155; font-size:13px;'>[{rel.type}]</b><br><span style='color:#64748B; font-size:11px;'>{rel.source} &rarr; {rel.target}</span>"
        if prop_lines:
            title_html += f"<hr style='margin:4px 0;'><table style='font-size:11px;'>{prop_lines}</table>"
        title_html += "</div>"
        
        edges_data.append({
            "id": rid,
            "from": rel.source,
            "to": rel.target,
            "label": f"  {rel.type}  ",
            "title": title_html,
            "color": {"color": edge_color, "highlight": "#F59E0B", "hover": "#64748B"},
            "width": 2 if not is_matched else 3,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
            "font": {"size": 10, "color": "#475569", "face": "Arial, sans-serif", "background": "rgba(255,255,255,0.9)", "strokeWidth": 0, "align": "middle"},
            "smooth": {"type": "continuous", "roundness": 0.15}
        })

    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
            body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
            #mynetwork {{
                width: 100%;
                height: 680px;
                border: 1px solid #cfd8d5;
                background-color: #fbfaf6;
                border-radius: 2px;
            }}
            .vis-tooltip {{
                background-color: white !important;
                border: 1px solid #CBD5E1 !important;
                border-radius: 2px !important;
                box-shadow: 0 4px 12px rgba(32, 37, 43, 0.12) !important;
                color: #334155 !important;
                padding: 0 !important;
                pointer-events: none;
            }}
            #controls {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                z-index: 100;
                display: flex;
                gap: 8px;
                background: rgba(255,255,255,0.9);
                padding: 6px;
                border-radius: 2px;
                box-shadow: 0 2px 8px rgba(32,37,43,0.1);
                border: 1px solid #cfd8d5;
            }}
            .ctrl-btn {{
                background: white; border: 1px solid #CBD5E1; border-radius: 4px; padding: 6px 10px; cursor: pointer; font-size: 14px; color: #475569; font-weight: 500;
                transition: all 0.2s;
            }}
            .ctrl-btn:hover {{ background: #F1F5F9; border-color: #94A3B8; color: #0F172A; }}
            
            #legend {{
                position: absolute;
                top: 20px;
                right: 20px;
                z-index: 100;
                background: rgba(255,255,255,0.95);
                padding: 12px;
                border-radius: 2px;
                box-shadow: 0 2px 8px rgba(32,37,43,0.1);
                border: 1px solid #cfd8d5;
                max-width: 180px;
                font-size: 12px;
                max-height: 560px;
                overflow-y: auto;
            }}
            .legend-title {{ font-weight: 600; margin-bottom: 8px; color: #1E293B; border-bottom: 1px solid #E2E8F0; padding-bottom: 4px; }}
            .legend-item {{ display: flex; align-items: center; margin-bottom: 6px; justify-content: space-between; }}
            .legend-label-group {{ display: flex; align-items: center; gap: 6px; }}
            .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; border: 1px solid rgba(0,0,0,0.1); }}
            .badge {{ background: #e7efed; color: #1e5b61; padding: 2px 6px; border-radius: 2px; font-size: 10px; font-weight: 600; }}
            .rel-badge {{ background: #fbfaf6; border: 1px solid #cfd8d5; color: #47545a; padding: 2px 6px; border-radius: 2px; font-size: 10px; font-weight: 600; }}
        </style>
    </head>
    <body>
        <div style="position: relative; width: 100%;">
            <div id="mynetwork"></div>
            
            <div id="legend">
                <div class="legend-title">Node Labels</div>
                <div id="node-legend-container"></div>
                <div class="legend-title" style="margin-top: 12px;">Relationship Types</div>
                <div id="rel-legend-container"></div>
            </div>
            
            <div id="controls">
                <button class="ctrl-btn" onclick="network.fit({{animation: true}})" title="Fit to View">Fit view</button>
                <button class="ctrl-btn" onclick="zoom(0.2)" title="Zoom In">Zoom in</button>
                <button class="ctrl-btn" onclick="zoom(-0.2)" title="Zoom Out">Zoom out</button>
                <button class="ctrl-btn" id="physics-btn" onclick="togglePhysics()" title="Toggle Physics">⚡ Freeze</button>
            </div>
        </div>

        <script type="text/javascript">
            var nodes = new vis.DataSet({nodes_json});
            var edges = new vis.DataSet({edges_json});

            var container = document.getElementById('mynetwork');
            var data = {{ nodes: nodes, edges: edges }};
            var options = {{
                physics: {{
                    enabled: true,
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: {{
                        gravitationalConstant: -60,
                        centralGravity: 0.015,
                        springLength: 120,
                        springConstant: 0.08,
                        damping: 0.4,
                        avoidOverlap: 0.5
                    }},
                    stabilization: {{ iterations: 150 }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 100,
                    zoomView: true,
                    dragView: true,
                    dragNodes: true
                }},
                layout: {{
                    improvedLayout: true
                }}
            }};
            var network = new vis.Network(container, data, options);
            
            // Build Legend dynamically
            const nodeLegendContainer = document.getElementById('node-legend-container');
            const relLegendContainer = document.getElementById('rel-legend-container');
            
            // Aggregate labels
            const labelCounts = {{}};
            const labelColors = {{}};
            nodes.forEach(n => {{
                let lbl = n.label.split("<i>:")[1]?.split("</i>")[0] || "Entity";
                labelCounts[lbl] = (labelCounts[lbl] || 0) + 1;
                labelColors[lbl] = n.color.background;
            }});
            
            const relCounts = {{}};
            edges.forEach(e => {{
                let type = e.label.trim();
                relCounts[type] = (relCounts[type] || 0) + 1;
            }});
            
            Object.keys(labelCounts).sort().forEach(lbl => {{
                let div = document.createElement('div');
                div.className = 'legend-item';
                div.innerHTML = `<div class="legend-label-group"><span class="dot" style="background-color: ${{labelColors[lbl]}};"></span> <span style="color:#334155;">${{lbl}}</span></div> <span class="badge">${{labelCounts[lbl]}}</span>`;
                nodeLegendContainer.appendChild(div);
            }});
            
            Object.keys(relCounts).sort().forEach(type => {{
                let div = document.createElement('div');
                div.className = 'legend-item';
                div.innerHTML = `<span class="rel-badge">${{type}}</span> <span class="badge">${{relCounts[type]}}</span>`;
                relLegendContainer.appendChild(div);
            }});
            
            function zoom(scale) {{
                var newScale = network.getScale() * (1 + scale);
                network.moveTo({{ scale: newScale, animation: {{ duration: 300 }} }});
            }}
            
            var physicsEnabled = true;
            function togglePhysics() {{
                physicsEnabled = !physicsEnabled;
                network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
                document.getElementById('physics-btn').innerHTML = physicsEnabled ? '⚡ Freeze' : '▶️ Unfreeze';
            }}
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=700)


# ======================================================================================
# 5. SECTION RENDERERS

# ======================================================================================

def go_to_simulation():
    """Route the cover action through the existing sidebar navigation state."""
    st.session_state["requested_section"] = "02  Simulation"


def render_experiment_cover():
    """Renders the opening spread of the digital practical manual."""
    st.markdown(f"""
        <section class="experiment-cover">
            <div class="cover-copy">
                <div class="hero-eyebrow"><span class="hero-dot"></span>VIRTUAL LABORATORY / DATABASE MANAGEMENT SYSTEMS <span class="hero-badge">{EXPERIMENT_CONFIG['lab_code']}</span></div>
                <div class="manual-kicker">Experiment 08 / Database Management Systems</div>
                <h2>Create and Manage<br>a Graph Database</h2>
                <p>Explore how connected data is represented, queried, and visualized through the labeled property graph model.</p>
                <div class="cover-meta"><span>Interactive simulation</span><span>Cypher engine</span><span>Practical record</span></div>
            </div>
            <div class="cover-visual" aria-label="Abstract graph showing connected nodes">
                <svg viewBox="0 0 430 270" role="img" aria-hidden="true">
                    <path class="graph-line" d="M72 150 L165 72 L286 104 L360 194 L215 222 L72 150 M165 72 L215 222 M286 104 L215 222" />
                    <path class="graph-line faint" d="M165 72 L360 194 M72 150 L286 104" />
                    <circle class="graph-node node-teal" cx="72" cy="150" r="19" />
                    <circle class="graph-node node-ink" cx="165" cy="72" r="24" />
                    <circle class="graph-node node-warm" cx="286" cy="104" r="17" />
                    <circle class="graph-node node-teal" cx="360" cy="194" r="22" />
                    <circle class="graph-node node-ink" cx="215" cy="222" r="29" />
                    <text x="41" y="188">Student</text><text x="139" y="39">Course</text>
                    <text x="270" y="78">Faculty</text><text x="334" y="236">Graph</text>
                </svg>
            </div>
        </section>
    """, unsafe_allow_html=True)
    st.button("Begin experiment", type="primary", on_click=go_to_simulation, key="begin_experiment")


def render_graph_concept_diagram():
    """Adds one compact visual explanation to the textbook introduction."""
    st.markdown("""
        <div class="concept-diagram">
            <div><span class="diagram-node">NODE</span><small>entity</small></div>
            <div class="diagram-arrow">→<small>relationship</small></div>
            <div><span class="diagram-node accent">NODE</span><small>entity</small></div>
            <div class="diagram-arrow">→<small>traversal</small></div>
            <div><span class="diagram-node warm">NODE</span><small>entity</small></div>
        </div>
    """, unsafe_allow_html=True)


def render_theory_section():
    """Renders Section 1: Theory, Background, Architecture, Cypher, and Procedure."""
    st.markdown('<div class="manual-kicker">01 / Theoretical Framework</div>', unsafe_allow_html=True)
    st.header("Graph Databases")
    st.markdown('<div class="manual-intro"><strong>Aim</strong><br>' + THEORY_CONTENT["aim"] + '</div>', unsafe_allow_html=True)

    st.subheader("Learning Objectives")
    for i, obj in enumerate(THEORY_CONTENT["learning_objectives"]):
        st.write(f"{i + 1}. {obj}")

    st.divider()

    # Create tabs for better organization
    tabs = st.tabs([
        "Introduction", 
        "RDBMS vs Graph", 
        "Graph Architecture", 
        "Cypher & CRUD", 
        "Procedure", 
        "Glossary"
    ])

    with tabs[0]:
        render_graph_concept_diagram()
        st.markdown(THEORY_CONTENT["introduction"])
    
    with tabs[1]:
        st.markdown(THEORY_CONTENT["rdbms_vs_graph"])
        st.markdown('<div class="manual-intro"><strong>Key takeaway</strong><br>Index-Free Adjacency (IFA) keeps graph traversal proportional to the path being explored, rather than the total database size.</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown(THEORY_CONTENT["neo4j_architecture"])

    with tabs[3]:
        st.markdown(THEORY_CONTENT["cypher_crud"])

    with tabs[4]:
        st.markdown(THEORY_CONTENT["setup_procedure"])
        st.divider()
        st.subheader("Step-by-Step Experimental Procedure")
        for step in THEORY_CONTENT["procedure"]:
            st.write(f"- {step}")
        st.divider()
        st.markdown('<div class="manual-label">Precautions and Common Mistakes</div>', unsafe_allow_html=True)
        st.markdown(THEORY_CONTENT["precautions"])

    with tabs[5]:
        st.markdown("### Comprehensive Key Terminology Reference")
        glossary_df = pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Term", "Formal Definition & Operational Role"]
        )
        st.dataframe(glossary_df, use_container_width=True, hide_index=True)


def render_simulation_section():
    """Renders Section 2: Interactive Graph Sandbox, Visual UI Controls, Cypher Console, Visualizer & Logger."""
    st.markdown('<div class="manual-kicker">02 / Laboratory Workstation</div>', unsafe_allow_html=True)
    st.header("Interactive Simulation")
    st.markdown('<div class="manual-intro">Create entities, establish directed relationships, and execute Cypher against the embedded graph engine. Observe each change in the topology below.</div>', unsafe_allow_html=True)

    graph: PropertyGraph = st.session_state["graph"]
    engine: CypherEngine = st.session_state["cypher_engine"]

    preset_options = [
        "University Academic Knowledge Graph (Default)",
        "Social Network & Friendships",
        "Financial Fraud Detection Ring",
        "Blank / Empty Graph"
    ]
    current_preset_idx = st.session_state.get("preset_index", 0)
    active_preset_name = st.session_state.get("active_preset", preset_options[current_preset_idx])
    schema = PRESET_SCHEMAS.get(active_preset_name, PRESET_SCHEMAS["University Academic Knowledge Graph (Default)"])
    metrics = graph.get_metrics()

    # ----------------------------------------------------------------------------------
    # TOP CONTROL BAR: Grouped Presets, Live Metrics & Reset
    # ----------------------------------------------------------------------------------
    with st.container(border=True):
        col_preset1, col_preset2, col_metrics, col_reset = st.columns([2.2, 1.2, 2.2, 1.0])

        with col_preset1:
            preset_choice = st.selectbox(
                "Load Domain Graph Preset:",
                options=preset_options,
                index=current_preset_idx,
                key="simulation_preset_select"
            )

        with col_preset2:
            st.write("")
            st.write("")
            if st.button("Load Selected Preset", use_container_width=True):
                if preset_choice.startswith("University"):
                    graph.load_university_graph()
                    st.session_state["preset_index"] = 0
                elif preset_choice.startswith("Social"):
                    graph.load_social_graph()
                    st.session_state["preset_index"] = 1
                elif preset_choice.startswith("Financial"):
                    graph.load_fraud_graph()
                    st.session_state["preset_index"] = 2
                else:
                    graph.clear()
                    st.session_state["preset_index"] = 3

                st.session_state["active_preset"] = preset_choice
                st.session_state["matched_node_ids"] = []
                st.session_state["matched_rel_ids"] = []
                st.session_state["last_cypher_result"] = None
                # Reset widget keys so new preset defaults apply
                for k in ["node_label_sel", "custom_label_inp", "node_id_inp", "node_name_inp", 
                          "prop_key1_inp", "prop_val1_inp", "prop_key2_inp", "prop_val2_inp",
                          "rel_type_sel", "custom_rel_inp", "rel_prop_k_inp", "rel_prop_v_inp"]:
                    st.session_state.pop(k, None)
                st.toast(f"Loaded '{preset_choice}' successfully!")
                st.rerun()

        with col_metrics:
            st.markdown('<div class="manual-label" style="margin-bottom:6px;">Current Graph Record</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-chips-row">'
                f'<span class="metric-chip"><strong>{metrics["num_nodes"]}</strong> nodes</span>'
                f'<span class="metric-chip"><strong>{metrics["num_relationships"]}</strong> relationships</span>'
                f'<span class="metric-chip"><strong>{metrics["num_labels"]}</strong> labels</span>'
                f'<span class="metric-chip"><strong>{metrics["density"]}</strong> density</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        with col_reset:
            st.write("")
            st.write("")
            if st.button("Reset Graph", type="secondary", use_container_width=True):
                graph.clear()
                st.session_state["matched_node_ids"] = []
                st.session_state["matched_rel_ids"] = []
                st.session_state["last_cypher_result"] = None
                st.toast("Graph cleared.")
                st.rerun()

    # ----------------------------------------------------------------------------------
    # CENTERPIECE: VISUAL GRAPH CANVAS (Main Focus of Virtual Lab)
    # ----------------------------------------------------------------------------------
    st.write("")
    with st.container(border=True):
        top_col1, top_col2, top_col3 = st.columns([2.8, 2.4, 1.2])
        with top_col1:
            st.subheader("Topology and Relationships")
        with top_col2:
            st.write("")
            view_mode = st.radio(
                "Graph Rendering Engine:",
                ["Interactive Graph Canvas (Recommended)", "Static Plotly Layout"],
                horizontal=True,
                label_visibility="collapsed"
            )
        with top_col3:
            st.write("")
            if st.button("Clear Highlighting", use_container_width=True):
                st.session_state["matched_node_ids"] = []
                st.session_state["matched_rel_ids"] = []
                st.rerun()

        if view_mode == "Interactive Graph Canvas (Recommended)":
            render_interactive_graph_canvas(
                graph=graph,
                matched_node_ids=st.session_state.get("matched_node_ids"),
                matched_rel_ids=st.session_state.get("matched_rel_ids")
            )
        else:
            pc1, pc2 = st.columns(2)
            with pc1:
                layout_opt = st.selectbox("Plotly Layout:", ["Spring (Force-Directed)", "Kamada-Kawai", "Circular", "Shell"], index=0)
            with pc2:
                label_opt = st.selectbox("Node Display:", ["Name / Label", "Name Only", "Node ID", "Label Only"], index=0)
            fig = render_graph_figure(
                graph=graph,
                matched_node_ids=st.session_state.get("matched_node_ids"),
                matched_rel_ids=st.session_state.get("matched_rel_ids"),
                layout_algorithm=layout_opt,
                node_label_mode=label_opt
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # ----------------------------------------------------------------------------------
    # WORKSTATION INTERACTIVE CONTROLS (Grouped cleanly in tabs)
    # ----------------------------------------------------------------------------------
    st.write("")
    tab_builder, tab_cypher, tab_logger = st.tabs([
        "Visual Graph Builder (UI Controls)",
        "Cypher Query Editor & Console",
        "Observation Sheet"
    ])

    # ----------------------------------------------------------------------------------
    # TAB 1: VISUAL GRAPH BUILDER (UI CONTROLS - CRUD FOR ALL USE CASES)
    # ----------------------------------------------------------------------------------
    with tab_builder:
        st.subheader("Visual Graph Builder")
        st.caption("Construct and manipulate nodes, labels, relationships, and properties using intuitive graphical controls.")

        sub_tab_node, sub_tab_rel, sub_tab_set, sub_tab_del = st.tabs([
            "Add Node",
            "Add Relationship",
            "Update Property (SET)",
            "Delete Entity"
        ])

        # SUB-TAB A: ADD NODE (CREATE)
        with sub_tab_node:
            col_lbl, col_id, col_name = st.columns(3)
            with col_lbl:
                label_options = schema["node_labels"]
                def_lbl_idx = 0
                if "node_label_sel" in st.session_state and st.session_state["node_label_sel"] in label_options:
                    def_lbl_idx = label_options.index(st.session_state["node_label_sel"])
                node_label = st.selectbox("Entity Label (Type):", label_options, index=def_lbl_idx, key="node_label_sel")
                if node_label == "Custom...":
                    custom_lbl_val = st.session_state.get("custom_label_inp", "Topic")
                    custom_node_label = st.text_input("Custom Label Name:", value=custom_lbl_val, key="custom_label_inp")
                    active_label_for_node = custom_node_label.strip() if custom_node_label.strip() else "Entity"
                else:
                    active_label_for_node = node_label

            lbl_defaults = schema.get("default_properties", {}).get(node_label, {
                "id_prefix": "node_", "name": "New Entity", "key1": "code", "val1": "E101", "key2": "status", "val2": "Active"
            })

            with col_id:
                def_id_val = st.session_state.get("node_id_inp", f"{lbl_defaults.get('id_prefix', 'node_')}{len(graph.nodes) + 1}")
                node_id_input = st.text_input("Node ID / Key:", value=def_id_val, key="node_id_inp", help="Unique identifier for the node")
            with col_name:
                def_name_val = st.session_state.get("node_name_inp", lbl_defaults.get("name", "New Entity"))
                node_name_input = st.text_input("Display Name:", value=def_name_val, key="node_name_inp")

            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                def_k1_val = st.session_state.get("prop_key1_inp", lbl_defaults.get("key1", "code"))
                def_v1_val = st.session_state.get("prop_val1_inp", lbl_defaults.get("val1", "VAL1"))
                prop_key1 = st.text_input("Property 1 Key:", value=def_k1_val, key="prop_key1_inp")
                prop_val1 = st.text_input("Property 1 Value:", value=def_v1_val, key="prop_val1_inp")
            with col_p2:
                def_k2_val = st.session_state.get("prop_key2_inp", lbl_defaults.get("key2", "status"))
                def_v2_val = st.session_state.get("prop_val2_inp", lbl_defaults.get("val2", "Active"))
                prop_key2 = st.text_input("Property 2 Key:", value=def_k2_val, key="prop_key2_inp")
                prop_val2 = st.text_input("Property 2 Value:", value=def_v2_val, key="prop_val2_inp")
            with col_p3:
                st.write("")
                st.write("")
                if st.button("Create Node", type="primary", use_container_width=True):
                    try:
                        props = {"name": node_name_input}
                        if prop_key1 and prop_val1:
                            try:
                                props[prop_key1] = float(prop_val1) if "." in prop_val1 else int(prop_val1)
                            except ValueError:
                                props[prop_key1] = prop_val1
                        if prop_key2 and prop_val2:
                            try:
                                props[prop_key2] = float(prop_val2) if "." in prop_val2 else int(prop_val2)
                            except ValueError:
                                props[prop_key2] = prop_val2

                        graph.add_node(node_id_input, [active_label_for_node], props)
                        st.session_state["matched_node_ids"] = [node_id_input]

                        # Log trial
                        st.session_state["trials"].append({
                            "Trial #": len(st.session_state["trials"]) + 1,
                            "Operation": "CREATE Node",
                            "Query / Action": f"CREATE (:{active_label_for_node} {{id: '{node_id_input}', name: '{node_name_input}'}})",
                            "Result": f"Node '{node_id_input}' added",
                            "Status": "Success",
                            "Timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        st.toast(f"Node '{node_id_input}' created successfully!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Error creating node: {str(ex)}")

        # SUB-TAB B: ADD RELATIONSHIP (CREATE)
        with sub_tab_rel:
            node_options = [f"{nid} ({':'.join(n.labels)}: {n.display_name()})" for nid, n in graph.nodes.items()]
            if len(node_options) < 2:
                st.warning("Please create at least 2 nodes before creating a relationship.")
            else:
                col_src, col_rel, col_tgt = st.columns(3)
                with col_src:
                    src_choice = st.selectbox("Source Node (From):", options=node_options, index=0)
                    src_id = src_choice.split(" ")[0]
                with col_rel:
                    rel_options = schema["rel_types"]
                    def_rel_idx = 0
                    if "rel_type_sel" in st.session_state and st.session_state["rel_type_sel"] in rel_options:
                        def_rel_idx = rel_options.index(st.session_state["rel_type_sel"])
                    rel_type = st.selectbox("Relationship Type:", rel_options, index=def_rel_idx, key="rel_type_sel")
                    if rel_type == "Custom...":
                        custom_rel_val = st.session_state.get("custom_rel_inp", "CONNECTED_TO")
                        custom_rel_input = st.text_input("Custom Relationship Type:", value=custom_rel_val, key="custom_rel_inp")
                        active_rel_type = custom_rel_input.strip().upper() if custom_rel_input.strip() else "CONNECTED_TO"
                    else:
                        active_rel_type = rel_type

                with col_tgt:
                    tgt_choice = st.selectbox("Target Node (To):", options=node_options, index=min(1, len(node_options) - 1))
                    tgt_id = tgt_choice.split(" ")[0]

                rel_defaults = schema.get("default_rel_properties", {}).get(rel_type, {"key": "weight", "val": "1.0"})
                col_rp1, col_rp2, col_rp3 = st.columns(3)
                with col_rp1:
                    def_rk = st.session_state.get("rel_prop_k_inp", rel_defaults.get("key", "weight"))
                    r_prop_k = st.text_input("Rel Property Key:", value=def_rk, key="rel_prop_k_inp")
                with col_rp2:
                    def_rv = st.session_state.get("rel_prop_v_inp", rel_defaults.get("val", "1.0"))
                    r_prop_v = st.text_input("Rel Property Value:", value=def_rv, key="rel_prop_v_inp")
                with col_rp3:
                    st.write("")
                    st.write("")
                    if st.button("Create Relationship", type="primary", use_container_width=True):
                        try:
                            r_props = {}
                            if r_prop_k and r_prop_v:
                                try:
                                    r_props[r_prop_k] = float(r_prop_v) if "." in r_prop_v else int(r_prop_v)
                                except ValueError:
                                    r_props[r_prop_k] = r_prop_v
                            r = graph.add_relationship(src_id, tgt_id, active_rel_type, r_props)
                            st.session_state["matched_node_ids"] = [src_id, tgt_id]
                            st.session_state["matched_rel_ids"] = [r.id]

                            # Log trial
                            st.session_state["trials"].append({
                                "Trial #": len(st.session_state["trials"]) + 1,
                                "Operation": "CREATE Rel",
                                "Query / Action": f"CREATE ({src_id})-[:{active_rel_type}]->({tgt_id})",
                                "Result": f"Rel '{r.id}' added",
                                "Status": "Success",
                                "Timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            st.toast(f"Relationship '{active_rel_type}' created!")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Error creating relationship: {str(ex)}")

        # SUB-TAB C: UPDATE PROPERTY (SET)
        with sub_tab_set:
            if not graph.nodes:
                st.info("No nodes available to update.")
            else:
                col_u1, col_u2, col_u3, col_u4 = st.columns([2.2, 1.6, 1.6, 1.2])
                with col_u1:
                    u_node_choice = st.selectbox("Select Target Node:", options=node_options, key="update_node_sel")
                    u_nid = u_node_choice.split(" ")[0]
                    selected_node = graph.nodes.get(u_nid)
                    if selected_node and selected_node.properties:
                        props_preview = " · ".join([f"**{k}**: {v}" for k, v in selected_node.properties.items()])
                        st.caption(f"Current properties: {props_preview}")

                with col_u2:
                    existing_keys = [k for k in selected_node.properties.keys() if k != "name"] if selected_node else []
                    prop_key_options = existing_keys + ["name", "Custom Key..."]
                    prop_choice = st.selectbox("Property Key to Update:", options=prop_key_options, key=f"prop_key_choice_{u_nid}")
                    if prop_choice == "Custom Key...":
                        set_k = st.text_input("Enter Property Name:", value="status", key=f"custom_set_key_{u_nid}")
                    else:
                        set_k = prop_choice

                with col_u3:
                    current_val = str(selected_node.properties.get(set_k, "")) if selected_node else ""
                    set_v = st.text_input("New Property Value:", value=current_val, key=f"set_val_{u_nid}_{set_k}")

                with col_u4:
                    st.write("")
                    st.write("")
                    if st.button("SET Property", type="primary", use_container_width=True):
                        try:
                            try:
                                v_parsed = float(set_v) if "." in set_v else int(set_v)
                            except ValueError:
                                v_parsed = set_v
                            succ, msg = graph.update_node(u_nid, {set_k: v_parsed})
                            st.session_state["matched_node_ids"] = [u_nid]
                            st.session_state["trials"].append({
                                "Trial #": len(st.session_state["trials"]) + 1,
                                "Operation": "SET Property",
                                "Query / Action": f"MATCH ({u_nid}) SET {set_k} = {set_v}",
                                "Result": msg[:40],
                                "Status": "Success" if succ else "Failed",
                                "Timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            st.toast(msg)
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Error updating node: {str(ex)}")

        # SUB-TAB D: DELETE ENTITY
        with sub_tab_del:
            col_dt, col_dt_pick, col_dt_btn = st.columns([1.5, 2.5, 1.5])
            with col_dt:
                del_mode = st.radio("Entity to Delete:", ["Node", "Relationship"])
            with col_dt_pick:
                if del_mode == "Node":
                    del_node_choice = st.selectbox("Select Node:", options=node_options, key="del_n_choice")
                    del_id = del_node_choice.split(" ")[0] if del_node_choice else ""
                    detach_flag = st.checkbox("DETACH DELETE (Delete attached relationships)", value=True)
                else:
                    rel_options = [f"{rid} ({r.source} -[:{r.type}]-> {r.target})" for rid, r in graph.relationships.items()]
                    if not rel_options:
                        st.info("No relationships in graph.")
                        del_id = ""
                    else:
                        del_rel_choice = st.selectbox("Select Relationship:", options=rel_options, key="del_r_choice")
                        del_id = del_rel_choice.split(" ")[0]
                    detach_flag = False

            with col_dt_btn:
                st.write("")
                st.write("")
                if st.button("Execute Delete", type="secondary", use_container_width=True):
                    if del_mode == "Node":
                        succ, msg = graph.delete_node(del_id, detach=detach_flag)
                    else:
                        succ, msg = graph.delete_relationship(del_id)

                    st.session_state["trials"].append({
                        "Trial #": len(st.session_state["trials"]) + 1,
                        "Operation": "DELETE",
                        "Query / Action": f"DETACH DELETE {del_id}" if detach_flag else f"DELETE {del_id}",
                        "Result": msg[:40],
                        "Status": "Success" if succ else "Failed",
                        "Timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    if succ:
                        st.session_state["matched_node_ids"] = []
                        st.session_state["matched_rel_ids"] = []
                        st.toast(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    # ----------------------------------------------------------------------------------
    # TAB 2: CYPHER QUERY CONSOLE
    # ----------------------------------------------------------------------------------
    with tab_cypher:
        st.subheader("Cypher Query Editor & Console")
        st.caption("Execute declarative Cypher queries against the in-memory graph. Supported: MATCH, CREATE, SET, DELETE, DETACH DELETE, WHERE, aggregations.")

        cypher_examples = schema["cypher_examples"]
        col_ex, col_load = st.columns([3.5, 1.2])
        with col_ex:
            selected_example = st.selectbox(
                "Predefined Cypher Examples:",
                options=list(cypher_examples.keys()),
                index=0,
                key=f"cypher_ex_sel_{active_preset_name}"
            )

        with col_load:
            st.write("")
            st.write("")
            if st.button("Load Query", use_container_width=True):
                if cypher_examples.get(selected_example):
                    st.session_state["current_query_input"] = cypher_examples[selected_example]
                    st.rerun()

        default_cypher = list(cypher_examples.values())[1] if len(cypher_examples) > 1 else "MATCH (n) RETURN n"
        query_input = st.text_area(
            "Enter Cypher Statement:",
            value=st.session_state.get("current_query_input", default_cypher),
            height=90,
            help="Type Cypher query. Examples: MATCH (n) RETURN n | CREATE (n:Student {name: 'Alice'})"
        )
        st.session_state["current_query_input"] = query_input

        col_run, _ = st.columns([1.5, 3.5])
        with col_run:
            run_clicked = st.button("Execute Cypher Query", type="primary", use_container_width=True)

        if run_clicked:
            res = engine.execute(query_input)
            st.session_state["last_cypher_result"] = res
            st.session_state["matched_node_ids"] = res["matched_node_ids"]
            st.session_state["matched_rel_ids"] = res["matched_rel_ids"]

            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Operation": "Cypher Query",
                "Query / Action": query_input[:65],
                "Result": res["message"][:40],
                "Status": "Success" if res["success"] else "Failed",
                "Timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state["trials"].append(trial_record)

        # Render Query Execution Results
        last_res = st.session_state.get("last_cypher_result")
        if last_res:
            st.divider()
            if last_res["success"]:
                st.success(f"**Query Succeeded** ({last_res['execution_time_ms']} ms): {last_res['message']}")
                df = last_res["dataframe"]
                if not df.empty:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No tabular records returned by this statement.")
            else:
                st.error(f"**Query Failed** ({last_res['execution_time_ms']} ms): {last_res['message']}")
                st.info("Tip: Ensure your labels, relationship types, and node variables are formatted correctly.")

    # ----------------------------------------------------------------------------------
    # TAB 3: EXPERIMENTAL DATA LOGGER (OBSERVATION SHEET)
    # ----------------------------------------------------------------------------------
    with tab_logger:
        st.subheader("Observation Sheet")
        st.caption("Record parameters, queries, and graph behavior across trials for inclusion in your official lab report.")

        col_log1, col_log2 = st.columns([1.8, 3.2])

        with col_log1:
            st.caption("Capture current graph metrics and last operation into your session log table:")
            if st.button("Record Current State as Trial", type="primary", use_container_width=True):
                trial_record = {
                    "Trial #": len(st.session_state["trials"]) + 1,
                    "Operation": "Graph Snapshot",
                    "Query / Action": f"Nodes: {metrics['num_nodes']}, Rels: {metrics['num_relationships']}",
                    "Result": f"Density: {metrics['density']}, AvgDeg: {metrics['avg_degree']}",
                    "Status": "Recorded",
                    "Timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.session_state["trials"].append(trial_record)
                st.toast(f"Trial #{trial_record['Trial #']} successfully logged!")

            if st.button("Clear Logged Trials", use_container_width=True):
                st.session_state["trials"] = []
                st.toast("Trial log cleared.")

        with col_log2:
            if st.session_state["trials"]:
                df_trials = pd.DataFrame(st.session_state["trials"])
                st.dataframe(df_trials, use_container_width=True, hide_index=True)
                csv_data = df_trials.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Trials as CSV",
                    data=csv_data,
                    file_name="graph_db_trials.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No trials recorded yet. Perform graph operations or click 'Record Current State as Trial' to begin.")


def render_quiz_section():
    """Renders Section 3: Assessment Quiz with Self-Grading and Feedback."""
    st.markdown('<div class="manual-kicker">03 / Knowledge Check</div>', unsafe_allow_html=True)
    st.header("Practical Assessment")
    st.markdown('<div class="manual-intro">Test your understanding of graph databases, graph architecture, and Cypher query syntax.</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="manual-label">Assessment / {len(QUIZ_QUESTIONS)} questions</div>'
        f'<div class="quiz-progress"><span style="width: {100 / len(QUIZ_QUESTIONS):.2f}%"></span></div>',
        unsafe_allow_html=True
    )

    with st.form("graph_lab_quiz_form"):
        user_responses = {}
        for q in QUIZ_QUESTIONS:
            with st.container():
                st.markdown(f"**Question {q['id']}:** {q['question']}")
                selected = st.radio(
                    label=f"Options for Question {q['id']}:",
                    options=q["options"],
                    index=st.session_state["quiz_answers"].get(q["id"], 0),
                    key=f"quiz_radio_{q['id']}",
                    label_visibility="collapsed"
                )
                user_responses[q["id"]] = q["options"].index(selected)
            st.divider()
        
        st.write("")
        submitted = st.form_submit_button("Submit Quiz for Evaluation", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Evaluation Results and Feedback")
        for q in QUIZ_QUESTIONS:
            user_ans = user_responses.get(q["id"])
            correct_ans = q["answer_index"]
            with st.container():
                if user_ans == correct_ans:
                    score += 1
                    st.success(f"**Question {q['id']}: Correct!**", icon="✅")
                    st.caption(f"_{q['explanation']}_")
                else:
                    st.error(f"**Question {q['id']}: Incorrect.**", icon="❌")
                    st.write(f"Your answer: `{q['options'][user_ans]}`")
                    st.write(f"**Correct Answer:** `{q['options'][correct_ans]}`")
                    st.caption(f"**Explanation:** _{q['explanation']}_")

        st.session_state["quiz_score"] = score
        perc = (score / len(QUIZ_QUESTIONS)) * 100
        
        st.markdown('<div class="manual-label">Assessment Complete</div>', unsafe_allow_html=True)
        if perc == 100:
            st.success(f"Perfect score: {score} / {len(QUIZ_QUESTIONS)} ({perc:.0f}%). Your responses have been recorded.")
        elif perc >= 70:
            st.info(f"Assessment recorded: {score} / {len(QUIZ_QUESTIONS)} ({perc:.0f}%). Review the explanations below.")
        else:
            st.warning(f"Assessment recorded: {score} / {len(QUIZ_QUESTIONS)} ({perc:.0f}%). Revisit the theory and try again.")

    elif st.session_state.get("quiz_submitted", False):
        st.success(f"Assessment already submitted. Current score: **{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}**")


def render_report_section():
    """Renders Section 4: Dynamic Lab Report Generator with Guaranteed PDF Export."""
    st.markdown('<div class="manual-kicker">04 / Record Submission</div>', unsafe_allow_html=True)
    st.header("Experiment Report")
    st.markdown('<div class="manual-intro">Compile your student details, observations, experimental log, and assessment result into the practical record.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="manual-label">Student Information</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            student_name = st.text_input("Student Name", value=st.session_state["student_info"].get("name", "Student Name"))
        with col2:
            student_id = st.text_input("Student Roll / ID", value=st.session_state["student_info"].get("id", "21CS01"))
        with col3:
            lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["date"] = str(lab_date)

    with st.container(border=True):
        st.markdown('<div class="manual-label">Observations</div>', unsafe_allow_html=True)
        student_notes = st.text_area(
            "Enter your interpretation of results, observations, and conclusions:",
            value=st.session_state.get("student_notes", (
                "During the experiment, we successfully created, queried, and managed an interconnected property graph. "
                "Using Cypher pattern matching, relationships were traversed efficiently without relational multi-table joins. "
                "Referential constraints were verified when attempting plain DELETE on connected nodes, demonstrating the necessity "
                "of DETACH DELETE for safely removing graph entities."
            )),
            height=130
        )
        st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()
    graph_metrics = st.session_state["graph"].get_metrics()

    st.divider()
    
    with st.container(border=True):
        st.markdown('<div class="manual-label">Report Preview</div>', unsafe_allow_html=True)
        st.markdown(f"**Experiment:** {EXPERIMENT_CONFIG['title']} ({EXPERIMENT_CONFIG['lab_code']})")
        
        st.markdown(
            f'<div class="report-facts">'
            f'<div class="report-fact"><small>Student</small><strong>{student_name}</strong></div>'
            f'<div class="report-fact"><small>Roll / ID</small><strong>{student_id}</strong></div>'
            f'<div class="report-fact"><small>Graph nodes</small><strong>{graph_metrics["num_nodes"]}</strong></div>'
            f'<div class="report-fact"><small>Assessment</small><strong>{st.session_state.get("quiz_score", 0)} / {len(QUIZ_QUESTIONS)}</strong></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        if not trials_df.empty:
            st.dataframe(trials_df, hide_index=True, use_container_width=True)
        else:
            st.info("Note: You have not recorded any trials in the Simulation tab yet. Your report will indicate 0 trials.")

    # Generate PDF bytes and write file to disk
    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=student_notes,
        graph_metrics=graph_metrics
    )

    # Save to local files for guaranteed download
    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    with open("lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    with st.container(border=True):
        st.markdown('<div class="manual-label">Export</div>', unsafe_allow_html=True)
        st.caption("Your personalized PDF report is ready to be downloaded and submitted.")
        col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.link_button(
            "Open / Download PDF Document",
            url="/app/static/lab_report.pdf",
            type="primary",
            use_container_width=True
        )

    with col_btn2:
        st.download_button(
            label="Download lab_report.pdf",
            data=pdf_bytes,
            file_name="graph_lab_report.pdf",
            mime="application/pdf",
            key="stream_pdf_btn",
            use_container_width=True
        )


# ======================================================================================
# 6. MAIN ENTRYPOINT & NAVIGATION
# ======================================================================================

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "graph" not in st.session_state:
        g = PropertyGraph()
        g.load_university_graph()
        st.session_state["graph"] = g
    if "cypher_engine" not in st.session_state:
        st.session_state["cypher_engine"] = CypherEngine(st.session_state["graph"])
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "matched_node_ids" not in st.session_state:
        st.session_state["matched_node_ids"] = []
    if "matched_rel_ids" not in st.session_state:
        st.session_state["matched_rel_ids"] = []
    if "last_cypher_result" not in st.session_state:
        st.session_state["last_cypher_result"] = None
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "21CS01",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""
    if "preset_index" not in st.session_state:
        st.session_state["preset_index"] = 0
    if "active_preset" not in st.session_state:
        st.session_state["active_preset"] = "University Academic Knowledge Graph (Default)"


def get_app_styles() -> str:
    """Generates clean, theme-adaptive CSS that renders seamlessly in both Light and Dark modes."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --accent: #0284c7;
        --accent-soft: rgba(2, 132, 199, 0.10);
        --accent-hover: #0369a1;
        --violet: #6366f1;
        --warm: #ea580c;
        --card-bg: rgba(127, 127, 127, 0.05);
        --card-border: rgba(127, 127, 127, 0.18);
        --card-subtle: rgba(127, 127, 127, 0.08);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --accent: #22d3c7;
            --accent-soft: rgba(34, 211, 199, 0.12);
            --accent-hover: #14b8a6;
            --violet: #7c6cf6;
            --warm: #f0a36a;
            --card-bg: rgba(255, 255, 255, 0.04);
            --card-border: rgba(255, 255, 255, 0.12);
            --card-subtle: rgba(255, 255, 255, 0.07);
        }
    }

    /* Container & Layout */
    .main .block-container {
        max-width: 1440px;
        padding: 2.2rem 3.5rem 4.5rem;
    }
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em !important;
    }
    
    h1 { font-size: 2.15rem !important; line-height: 1.15 !important; }
    h2 { font-size: 1.55rem !important; margin-top: 1.8rem !important; }
    h3 { font-size: 1.15rem !important; }

    /* Virtual Lab Header */
    .vlab-header {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-top: 3px solid var(--accent);
        border-radius: 14px;
        padding: 1.8rem 2.2rem 1.6rem;
        margin-bottom: 2.2rem;
    }
    .vlab-header h1 {
        margin: 0;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    .hero-eyebrow { color: var(--accent); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; }
    .experiment-cover .hero-eyebrow { display: flex; align-items: center; gap: 0.55rem; flex-wrap: wrap; }
    .hero-dot { display: inline-block; width: 7px; height: 7px; margin-right: 0.55rem; border-radius: 50%; background: var(--accent); vertical-align: 1px; }
    .hero-title-row { display: flex; align-items: center; gap: 0.9rem; flex-wrap: wrap; margin-top: 0.75rem; }
    .hero-title-row h1 { margin: 0 !important; }
    .hero-badge { background: linear-gradient(135deg, var(--accent), var(--violet)); color: #FFFFFF; padding: 0.32rem 0.7rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; white-space: nowrap; }
    .vlab-header p.subtitle { opacity: 0.8; font-size: 0.96rem; margin: 0.65rem 0 0; }
    .vlab-header span.badge { background: var(--accent-soft); color: var(--accent); padding: 0.25rem 0.55rem; border-radius: 4px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; white-space: nowrap; margin-left: 0.75rem; vertical-align: middle; }

    .manual-kicker, .manual-label {
        color: var(--accent) !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.13em !important;
        text-transform: uppercase;
    }
    .manual-intro {
        border-left: 3px solid var(--accent);
        padding: 0.55rem 1rem;
        margin: 1.1rem 0 1.8rem;
        background: var(--accent-soft);
        border-radius: 0 8px 8px 0;
    }
    .experiment-cover {
        display: grid;
        grid-template-columns: minmax(0, 1.05fr) minmax(300px, 0.95fr);
        gap: 2rem;
        align-items: center;
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-top: 3px solid var(--accent);
        border-radius: 14px;
        padding: 2.2rem 2.5rem;
        margin: 1.6rem 0 1.8rem;
    }
    .experiment-cover h2 { font-size: clamp(2rem, 4vw, 3.2rem) !important; line-height: 1.05 !important; margin: 0.7rem 0 1rem !important; }
    .cover-copy p { max-width: 34rem; opacity: 0.82; font-size: 1.02rem; line-height: 1.65; }
    .cover-meta { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1.5rem; opacity: 0.75; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .cover-meta span { border-left: 2px solid var(--accent); padding-left: 0.55rem; }
    .cover-visual { min-height: 250px; display: grid; place-items: center; background: var(--card-subtle); border: 1px solid var(--card-border); border-radius: 12px; }
    .cover-visual svg { width: 100%; max-width: 430px; height: auto; }
    .graph-line { fill: none; stroke: var(--accent); stroke-width: 2; stroke-dasharray: 5 5; }
    .graph-line.faint { stroke: #93C5FD; }
    .graph-node { stroke-width: 4; }
    .graph-node.node-teal { fill: #0284C7; }
    .graph-node.node-warm { fill: #EA580C; }
    .cover-visual text { opacity: 0.75; font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.04em; }

    .concept-diagram { display: flex; align-items: center; justify-content: center; gap: 1.2rem; padding: 1.4rem 1rem; margin: 0.5rem 0 2rem; border: 1px solid var(--card-border); background: var(--card-bg); border-radius: 10px; }
    .concept-diagram > div { display: grid; gap: 0.35rem; justify-items: center; }
    .concept-diagram small { opacity: 0.75; font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .diagram-node { display: inline-flex; align-items: center; justify-content: center; width: 4.8rem; height: 2.8rem; border: 2px solid currentColor; background: var(--card-subtle); border-radius: 4px; font: 600 0.72rem 'IBM Plex Mono', monospace; }
    .diagram-node.accent { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
    .diagram-node.warm { border-color: var(--warm); color: var(--warm); background: rgba(234, 88, 12, 0.08); }
    .diagram-arrow { color: var(--accent); font-size: 1.5rem; }
    .diagram-arrow small { display: block; font-size: 0.6rem; text-align: center; opacity: 0.75; }

    .report-facts { display: grid; grid-template-columns: repeat(4, 1fr); border: 1px solid var(--card-border); border-radius: 10px; background: var(--card-bg); margin: 1.2rem 0 1.6rem; }
    .report-fact { padding: 0.85rem 1rem; border-right: 1px solid var(--card-border); }
    .report-fact:last-child { border-right: 0; }
    .report-fact small { display: block; opacity: 0.75; font-size: 0.67rem; letter-spacing: 0.1em; text-transform: uppercase; }
    .report-fact strong { display: block; margin-top: 0.35rem; color: var(--accent); font-size: 1.05rem; overflow-wrap: anywhere; }

    .quiz-progress { height: 5px; background: var(--card-border); border-radius: 3px; overflow: hidden; margin: 1.2rem 0 2rem; }
    .quiz-progress span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--violet)); }

    /* Simulation metric chips */
    .metric-chips-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 4px; }
    .metric-chip { background: var(--card-subtle); border: 1px solid var(--card-border); border-radius: 6px; padding: 4px 9px; font-size: 0.8rem; }
    .metric-chip strong { color: var(--accent); font-weight: 700; }

    /* Sidebar */
    .sidebar-brand { color: var(--accent); font-size: 1.1rem; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 0.65rem; }
    .sidebar-code { opacity: 0.75; font-size: 0.72rem; line-height: 1.55; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1.5rem; }
    .sidebar-stat { border: 1px solid var(--card-border); border-radius: 9px; background: var(--card-subtle); padding: 0.65rem 0.75rem; margin: 0.5rem 0; }
    .sidebar-stat-label { display: block; opacity: 0.75; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
    .sidebar-stat-value { display: block; color: var(--accent); font-size: 0.9rem; font-weight: 700; margin-top: 0.2rem; }

    /* Tabs & Code Areas */
    [data-testid="stTabs"] [role="tablist"] { gap: 0.5rem; border-bottom: 1px solid var(--card-border); }
    [data-testid="stTabs"] button[role="tab"] { border-radius: 8px 8px 0 0; padding: 0.65rem 0.8rem; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--accent); font-weight: 600; }
    [data-testid="stTextArea"] textarea { font-family: 'IBM Plex Mono', monospace !important; }
    </style>
    """


def main():
    st.set_page_config(
        page_title="Create and Manage a Graph Database - Virtual Lab",
        page_icon=None,
        layout="wide"
    )

    init_session_state()

    # Dynamic application theme
    st.markdown(get_app_styles(), unsafe_allow_html=True)

    # Navigation Sidebar
    st.sidebar.markdown('<div class="sidebar-brand">VIRTUAL LAB</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="sidebar-code">Database Management Systems<br>{EXPERIMENT_CONFIG["lab_code"]}</div>', unsafe_allow_html=True)

    navigation_options = ["01  Theory", "02  Simulation", "03  Quiz", "04  Experiment Report"]
    requested_section = st.session_state.get("requested_section", navigation_options[0])
    section = st.sidebar.radio(
        "Navigation",
        options=navigation_options,
        index=navigation_options.index(requested_section),
        label_visibility="collapsed"
    )
    st.session_state["requested_section"] = section

    st.sidebar.divider()
    st.sidebar.markdown('<div class="manual-label">Session Record</div>', unsafe_allow_html=True)
    quiz_status = "Done" if st.session_state.get("quiz_submitted", False) else "Pending"
    st.sidebar.markdown(f'<div class="sidebar-stat"><span class="sidebar-stat-label">Quiz status</span><span class="sidebar-stat-value">{quiz_status}</span></div>', unsafe_allow_html=True)
    if st.session_state.get("quiz_submitted", False):
        st.sidebar.markdown(f'<div class="sidebar-stat"><span class="sidebar-stat-label">Quiz score</span><span class="sidebar-stat-value">{st.session_state.get("quiz_score", 0)} / {len(QUIZ_QUESTIONS)}</span></div>', unsafe_allow_html=True)

    st.sidebar.markdown(f'<div class="sidebar-stat"><span class="sidebar-stat-label">Recorded trials</span><span class="sidebar-stat-value">{len(st.session_state.get("trials", []))}</span></div>', unsafe_allow_html=True)
    m = st.session_state["graph"].get_metrics()
    st.sidebar.markdown(f'<div class="sidebar-stat"><span class="sidebar-stat-label">Graph size</span><span class="sidebar-stat-value">{m["num_nodes"]} nodes · {m["num_relationships"]} rels</span></div>', unsafe_allow_html=True)

    # Section Dispatcher
    if section == "01  Theory":
        render_experiment_cover()
        render_theory_section()
    elif section == "02  Simulation":
        render_simulation_section()
    elif section == "03  Quiz":
        render_quiz_section()
    elif section == "04  Experiment Report":
        render_report_section()


if __name__ == "__main__":
    main()




