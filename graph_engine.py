"""
graph_engine.py
Embedded In-Memory Labeled Property Graph (LPG) and Cypher Execution Engine.
Designed for educational Neo4j Virtual Labs to demonstrate core graph database concepts
without requiring external Neo4j server installations.
"""

import re
import time
import uuid
from typing import Dict, List, Set, Any, Tuple, Optional
import networkx as nx
import pandas as pd


class Node:
    """Represents a Node in a Labeled Property Graph."""
    def __init__(self, node_id: str, labels: Optional[List[str]] = None, properties: Optional[Dict[str, Any]] = None):
        self.id = str(node_id).strip()
        self.labels = set(labels) if labels else set()
        self.properties = dict(properties) if properties else {}

    def get(self, prop_name: str, default: Any = None) -> Any:
        if prop_name == "id":
            return self.id
        return self.properties.get(prop_name, default)

    def to_dict(self) -> Dict[str, Any]:
        d = {"id": self.id, "labels": list(self.labels)}
        d.update(self.properties)
        return d

    def display_name(self) -> str:
        return str(self.properties.get("name", self.id))

    def __repr__(self):
        labels_str = ":" + ":".join(sorted(self.labels)) if self.labels else ""
        return f"Node({self.id}{labels_str} {self.properties})"


class Relationship:
    """Represents a Directed Relationship (Edge) in a Labeled Property Graph."""
    def __init__(self, rel_id: str, source: str, target: str, rel_type: str, properties: Optional[Dict[str, Any]] = None):
        self.id = str(rel_id).strip()
        self.source = str(source).strip()
        self.target = str(target).strip()
        self.type = str(rel_type).strip().upper()
        self.properties = dict(properties) if properties else {}

    def get(self, prop_name: str, default: Any = None) -> Any:
        if prop_name == "id":
            return self.id
        if prop_name == "type":
            return self.type
        return self.properties.get(prop_name, default)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "source": self.source,
            "type": self.type,
            "target": self.target
        }
        d.update(self.properties)
        return d

    def __repr__(self):
        return f"Relationship({self.source}-[:{self.type} {self.properties}]->{self.target})"


class PropertyGraph:
    """
    In-memory Labeled Property Graph storing Nodes and Relationships
    with Index-Free Adjacency emulation.
    """
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.relationships: Dict[str, Relationship] = {}
        # Index-Free Adjacency pointers:
        self.outgoing: Dict[str, List[str]] = {}  # node_id -> [rel_id, ...]
        self.incoming: Dict[str, List[str]] = {}  # node_id -> [rel_id, ...]
        self.label_index: Dict[str, Set[str]] = {}  # label -> set of node_ids

    def clear(self):
        self.nodes.clear()
        self.relationships.clear()
        self.outgoing.clear()
        self.incoming.clear()
        self.label_index.clear()

    def add_node(self, node_id: str, labels: Optional[List[str]] = None, properties: Optional[Dict[str, Any]] = None) -> Node:
        node_id = str(node_id).strip()
        if not node_id:
            raise ValueError("Node ID cannot be empty.")
        
        # If node exists, update labels and properties
        if node_id in self.nodes:
            existing = self.nodes[node_id]
            if labels:
                for lbl in labels:
                    existing.labels.add(lbl)
                    self.label_index.setdefault(lbl, set()).add(node_id)
            if properties:
                existing.properties.update(properties)
            return existing

        node = Node(node_id, labels, properties)
        self.nodes[node_id] = node
        self.outgoing[node_id] = []
        self.incoming[node_id] = []

        for lbl in node.labels:
            self.label_index.setdefault(lbl, set()).add(node_id)

        return node

    def add_relationship(self, source_id: str, target_id: str, rel_type: str, 
                         properties: Optional[Dict[str, Any]] = None, rel_id: Optional[str] = None) -> Relationship:
        source_id = str(source_id).strip()
        target_id = str(target_id).strip()
        rel_type = str(rel_type).strip().upper()

        if source_id not in self.nodes:
            raise ValueError(f"Source node '{source_id}' does not exist.")
        if target_id not in self.nodes:
            raise ValueError(f"Target node '{target_id}' does not exist.")
        if not rel_type:
            raise ValueError("Relationship type cannot be empty.")

        if not rel_id:
            rel_id = f"rel_{source_id}_{rel_type}_{target_id}_{len(self.relationships) + 1}"

        rel = Relationship(rel_id, source_id, target_id, rel_type, properties)
        self.relationships[rel_id] = rel
        self.outgoing[source_id].append(rel_id)
        self.incoming[target_id].append(rel_id)
        return rel

    def delete_node(self, node_id: str, detach: bool = False) -> Tuple[bool, str]:
        node_id = str(node_id).strip()
        if node_id not in self.nodes:
            return False, f"Node '{node_id}' does not exist."

        connected_rels = set(self.outgoing.get(node_id, []) + self.incoming.get(node_id, []))
        if connected_rels and not detach:
            return False, (
                f"Neo4j Referential Integrity Constraint Violated: Cannot delete node '{node_id}' "
                f"because it still has {len(connected_rels)} connected relationship(s). "
                f"Use 'DETACH DELETE' to remove attached relationships automatically."
            )

        # Delete all attached relationships
        for rel_id in list(connected_rels):
            self.delete_relationship(rel_id)

        # Remove from label index
        node = self.nodes[node_id]
        for lbl in node.labels:
            if lbl in self.label_index and node_id in self.label_index[lbl]:
                self.label_index[lbl].remove(node_id)

        del self.nodes[node_id]
        if node_id in self.outgoing:
            del self.outgoing[node_id]
        if node_id in self.incoming:
            del self.incoming[node_id]

        return True, f"Successfully deleted node '{node_id}' (and {len(connected_rels)} connected relationships)."

    def delete_relationship(self, rel_id: str) -> Tuple[bool, str]:
        rel_id = str(rel_id).strip()
        if rel_id not in self.relationships:
            return False, f"Relationship '{rel_id}' does not exist."

        rel = self.relationships[rel_id]
        if rel.source in self.outgoing and rel_id in self.outgoing[rel.source]:
            self.outgoing[rel.source].remove(rel_id)
        if rel.target in self.incoming and rel_id in self.incoming[rel.target]:
            self.incoming[rel.target].remove(rel_id)

        del self.relationships[rel_id]
        return True, f"Successfully deleted relationship '{rel_id}'."

    def update_node(self, node_id: str, properties: Dict[str, Any], labels_to_add: Optional[List[str]] = None) -> Tuple[bool, str]:
        node_id = str(node_id).strip()
        if node_id not in self.nodes:
            return False, f"Node '{node_id}' not found."
        node = self.nodes[node_id]
        node.properties.update(properties)
        if labels_to_add:
            for lbl in labels_to_add:
                node.labels.add(lbl)
                self.label_index.setdefault(lbl, set()).add(node_id)
        return True, f"Node '{node_id}' updated successfully."

    def get_metrics(self) -> Dict[str, Any]:
        num_nodes = len(self.nodes)
        num_rels = len(self.relationships)
        labels = set()
        for n in self.nodes.values():
            labels.update(n.labels)
        rel_types = set(r.type for r in self.relationships.values())

        # Network density: for directed graph D = E / (V * (V - 1))
        density = 0.0
        if num_nodes > 1:
            density = round(num_rels / (num_nodes * (num_nodes - 1)), 4)

        avg_degree = round((2 * num_rels / num_nodes), 2) if num_nodes > 0 else 0.0

        # Isolated nodes (nodes with 0 incoming and 0 outgoing)
        isolated = 0
        for nid in self.nodes:
            deg = len(self.outgoing.get(nid, [])) + len(self.incoming.get(nid, []))
            if deg == 0:
                isolated += 1

        return {
            "num_nodes": num_nodes,
            "num_relationships": num_rels,
            "num_labels": len(labels),
            "labels": sorted(list(labels)),
            "num_rel_types": len(rel_types),
            "rel_types": sorted(list(rel_types)),
            "density": density,
            "avg_degree": avg_degree,
            "isolated_nodes": isolated
        }

    def to_networkx(self) -> nx.MultiDiGraph:
        """Converts graph to NetworkX MultiDiGraph for layout computation."""
        G = nx.MultiDiGraph()
        for nid, node in self.nodes.items():
            G.add_node(nid, labels=list(node.labels), **node.properties)
        for rid, rel in self.relationships.items():
            G.add_edge(rel.source, rel.target, key=rid, type=rel.type, **rel.properties)
        return G

    # ----------------------------------------------------------------------------------
    # SAMPLE DATASETS
    # ----------------------------------------------------------------------------------
    def load_university_graph(self):
        """Loads a realistic University Academic Knowledge Graph."""
        self.clear()
        # Departments
        self.add_node("dept_cs", ["Department"], {"name": "Computer Science & Engineering", "code": "CSE", "building": "Takshashila"})
        self.add_node("dept_ee", ["Department"], {"name": "Electrical Engineering", "code": "EE", "building": "Ramanujan"})
        self.add_node("dept_ma", ["Department"], {"name": "Mathematics", "code": "MATH", "building": "Aryabhatta"})

        # Courses
        self.add_node("cs101", ["Course"], {"name": "Programming & Data Structures", "code": "CS101", "credits": 4, "level": "UG"})
        self.add_node("cs201", ["Course"], {"name": "Algorithms & Complexity", "code": "CS201", "credits": 4, "level": "UG"})
        self.add_node("cs301", ["Course"], {"name": "Database Management Systems", "code": "CS301", "credits": 3, "level": "UG"})
        self.add_node("ee101", ["Course"], {"name": "Basic Electrical Technology", "code": "EE101", "credits": 3, "level": "UG"})
        self.add_node("ma101", ["Course"], {"name": "Linear Algebra & Calculus", "code": "MA101", "credits": 4, "level": "UG"})

        # Faculty
        self.add_node("prof_sharma", ["Faculty"], {"name": "Prof. R. Sharma", "dept": "CSE", "role": "HOD", "experience_yrs": 18})
        self.add_node("prof_banerjee", ["Faculty"], {"name": "Prof. S. Banerjee", "dept": "CSE", "role": "Professor", "experience_yrs": 12})
        self.add_node("prof_gupta", ["Faculty"], {"name": "Prof. A. Gupta", "dept": "EE", "role": "Assoc. Professor", "experience_yrs": 8})

        # Students
        self.add_node("s_alice", ["Student"], {"name": "Alice Smith", "roll": "21CS01", "dept": "CSE", "gpa": 9.15, "year": 3})
        self.add_node("s_bob", ["Student"], {"name": "Bob Kumar", "roll": "21CS02", "dept": "CSE", "gpa": 8.40, "year": 3})
        self.add_node("s_charlie", ["Student"], {"name": "Charlie Roy", "roll": "22EE05", "dept": "EE", "gpa": 9.60, "year": 2})
        self.add_node("s_diana", ["Student"], {"name": "Diana Das", "roll": "22CS14", "dept": "CSE", "gpa": 8.85, "year": 2})
        self.add_node("s_evan", ["Student"], {"name": "Evan Sen", "roll": "23CS20", "dept": "CSE", "gpa": 7.90, "year": 1})

        # Relationships
        # Faculty -> Department (BELONGS_TO)
        self.add_relationship("prof_sharma", "dept_cs", "BELONGS_TO", {"since": 2006})
        self.add_relationship("prof_banerjee", "dept_cs", "BELONGS_TO", {"since": 2012})
        self.add_relationship("prof_gupta", "dept_ee", "BELONGS_TO", {"since": 2016})

        # Course -> Department (OFFERED_BY)
        self.add_relationship("cs101", "dept_cs", "OFFERED_BY", {"semester": "Autumn"})
        self.add_relationship("cs201", "dept_cs", "OFFERED_BY", {"semester": "Spring"})
        self.add_relationship("cs301", "dept_cs", "OFFERED_BY", {"semester": "Autumn"})
        self.add_relationship("ee101", "dept_ee", "OFFERED_BY", {"semester": "Autumn"})
        self.add_relationship("ma101", "dept_ma", "OFFERED_BY", {"semester": "Autumn"})

        # Faculty -> Course (TEACHES)
        self.add_relationship("prof_sharma", "cs301", "TEACHES", {"academic_year": "2024-25"})
        self.add_relationship("prof_banerjee", "cs101", "TEACHES", {"academic_year": "2024-25"})
        self.add_relationship("prof_banerjee", "cs201", "TEACHES", {"academic_year": "2024-25"})
        self.add_relationship("prof_gupta", "ee101", "TEACHES", {"academic_year": "2024-25"})

        # Student -> Course (ENROLLED_IN)
        self.add_relationship("s_alice", "cs301", "ENROLLED_IN", {"grade": "Ex", "semester": "5th"})
        self.add_relationship("s_alice", "cs201", "ENROLLED_IN", {"grade": "A", "semester": "4th"})
        self.add_relationship("s_bob", "cs301", "ENROLLED_IN", {"grade": "B", "semester": "5th"})
        self.add_relationship("s_bob", "cs201", "ENROLLED_IN", {"grade": "B", "semester": "4th"})
        self.add_relationship("s_charlie", "ee101", "ENROLLED_IN", {"grade": "Ex", "semester": "3rd"})
        self.add_relationship("s_charlie", "ma101", "ENROLLED_IN", {"grade": "Ex", "semester": "1st"})
        self.add_relationship("s_diana", "cs201", "ENROLLED_IN", {"grade": "A", "semester": "3rd"})
        self.add_relationship("s_evan", "cs101", "ENROLLED_IN", {"grade": "A", "semester": "1st"})

        # Prerequisite Chain: CS101 -> CS201 -> CS301
        self.add_relationship("cs101", "cs201", "PREREQUISITE_OF", {"mandatory": True})
        self.add_relationship("cs201", "cs301", "PREREQUISITE_OF", {"mandatory": True})

        # Advising / Mentorship: Faculty -> Student
        self.add_relationship("prof_sharma", "s_alice", "ADVISES", {"project": "Graph Databases in VLSI"})
        self.add_relationship("prof_banerjee", "s_bob", "ADVISES", {"project": "Distributed Cypher Optimization"})

    def load_social_graph(self):
        """Loads a Social Network graph with friendships, interests, and groups."""
        self.clear()
        self.add_node("p_alex", ["Person"], {"name": "Alex", "city": "Kharagpur", "age": 22})
        self.add_node("p_bella", ["Person"], {"name": "Bella", "city": "Kolkata", "age": 23})
        self.add_node("p_chris", ["Person"], {"name": "Chris", "city": "Delhi", "age": 24})
        self.add_node("p_dan", ["Person"], {"name": "Dan", "city": "Bengaluru", "age": 21})

        self.add_node("g_ai", ["Group"], {"name": "AI & Graph Researchers", "members_count": 150})
        self.add_node("g_robotics", ["Group"], {"name": "Autonomous Systems Club", "members_count": 80})

        self.add_node("i_neo4j", ["Topic"], {"name": "Neo4j & Cypher", "domain": "Databases"})
        self.add_node("i_python", ["Topic"], {"name": "Python Data Science", "domain": "Programming"})

        self.add_relationship("p_alex", "p_bella", "FRIENDS_WITH", {"since": 2022})
        self.add_relationship("p_bella", "p_chris", "FRIENDS_WITH", {"since": 2021})
        self.add_relationship("p_chris", "p_dan", "FRIENDS_WITH", {"since": 2023})
        self.add_relationship("p_alex", "p_dan", "FRIENDS_WITH", {"since": 2024})

        self.add_relationship("p_alex", "g_ai", "MEMBER_OF", {"role": "Lead"})
        self.add_relationship("p_bella", "g_ai", "MEMBER_OF", {"role": "Member"})
        self.add_relationship("p_chris", "g_robotics", "MEMBER_OF", {"role": "Admin"})

        self.add_relationship("p_alex", "i_neo4j", "INTERESTED_IN", {"level": "Advanced"})
        self.add_relationship("p_bella", "i_python", "INTERESTED_IN", {"level": "Intermediate"})
        self.add_relationship("p_chris", "i_neo4j", "INTERESTED_IN", {"level": "Beginner"})

    def load_fraud_graph(self):
        """Loads a Financial Fraud Detection / Shared Identifiers graph."""
        self.clear()
        self.add_node("acc_101", ["Account"], {"acc_no": "ACC-101", "owner": "John", "balance": 15000})
        self.add_node("acc_102", ["Account"], {"acc_no": "ACC-102", "owner": "Mike", "balance": 3200})
        self.add_node("acc_103", ["Account"], {"acc_no": "ACC-103", "owner": "Sarah", "balance": 78000})
        self.add_node("acc_104", ["Account"], {"acc_no": "ACC-104", "owner": "Victor", "balance": 500})

        self.add_node("dev_a", ["Device"], {"device_id": "DEV-MAC-8812", "os": "Android 14"})
        self.add_node("ip_x", ["IPAddress"], {"ip": "192.168.1.105", "city": "Mumbai"})

        # Shared Device & IP (Fraud Ring indicator)
        self.add_relationship("acc_101", "dev_a", "USED_DEVICE", {"last_login": "2024-09-01"})
        self.add_relationship("acc_102", "dev_a", "USED_DEVICE", {"last_login": "2024-09-02"})
        self.add_relationship("acc_104", "dev_a", "USED_DEVICE", {"last_login": "2024-09-03"})

        self.add_relationship("acc_101", "ip_x", "LOGGED_FROM", {"timestamp": "10:15"})
        self.add_relationship("acc_102", "ip_x", "LOGGED_FROM", {"timestamp": "10:20"})

        # Suspicious Circular Money Transfers
        self.add_relationship("acc_101", "acc_102", "TRANSFER", {"amount": 5000, "txn_id": "TX901"})
        self.add_relationship("acc_102", "acc_104", "TRANSFER", {"amount": 4800, "txn_id": "TX902"})
        self.add_relationship("acc_104", "acc_101", "TRANSFER", {"amount": 4700, "txn_id": "TX903"})


# ======================================================================================
# CYPHER QUERY ENGINE (IN-MEMORY EDUCATIONAL IMPLEMENTATION)
# ======================================================================================

class CypherEngine:
    """
    Parses and executes standard Cypher query patterns over PropertyGraph.
    Supported operations:
      - MATCH (pattern) [WHERE ...] RETURN ...
      - MATCH (pattern) SET ... [RETURN ...]
      - MATCH (pattern) DELETE / DETACH DELETE ...
      - CREATE (pattern) [RETURN ...]
    """
    def __init__(self, graph: PropertyGraph):
        self.graph = graph

    def execute(self, query_text: str) -> Dict[str, Any]:
        """
        Executes a Cypher query string.
        Returns a dict with:
          success: bool
          message: str
          dataframe: pd.DataFrame
          matched_node_ids: list[str]
          matched_rel_ids: list[str]
          affected_count: int
          execution_time_ms: float
        """
        start_time = time.perf_counter()
        query = query_text.strip()
        if not query:
            return {
                "success": False,
                "message": "Empty query provided.",
                "dataframe": pd.DataFrame(),
                "matched_node_ids": [],
                "matched_rel_ids": [],
                "affected_count": 0,
                "execution_time_ms": 0.0
            }

        # Normalize whitespace (preserve inside quotes)
        # Check primary clause
        upper_query = query.upper()

        try:
            if upper_query.startswith("CREATE"):
                result = self._execute_create(query)
            elif upper_query.startswith("MATCH"):
                result = self._execute_match(query)
            else:
                return {
                    "success": False,
                    "message": (
                        "Unsupported Cypher query starting clause. "
                        "Supported initial clauses are 'MATCH' and 'CREATE'."
                    ),
                    "dataframe": pd.DataFrame(),
                    "matched_node_ids": [],
                    "matched_rel_ids": [],
                    "affected_count": 0,
                    "execution_time_ms": round((time.perf_counter() - start_time) * 1000, 2)
                }

            result["execution_time_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
            return result

        except Exception as e:
            return {
                "success": False,
                "message": f"Cypher Execution Error: {str(e)}",
                "dataframe": pd.DataFrame(),
                "matched_node_ids": [],
                "matched_rel_ids": [],
                "affected_count": 0,
                "execution_time_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }

    # ----------------------------------------------------------------------------------
    # CREATE EXECUTION
    # ----------------------------------------------------------------------------------
    def _execute_create(self, query: str) -> Dict[str, Any]:
        # Handle RETURN clause if present
        return_clause = None
        rem_query = query
        m_ret = re.search(r"\bRETURN\b\s+(.*)$", query, flags=re.IGNORECASE)
        if m_ret:
            return_clause = m_ret.group(1).strip()
            rem_query = query[:m_ret.start()].strip()

        create_body = rem_query[6:].strip() # Strip 'CREATE'
        
        # Check if creating node or relationship
        # Pattern: (a:Label {props}) or (a:Label {props})-[:REL {props}]->(b:Label {props})
        created_nodes = []
        created_rels = []
        node_var_map = {}

        # Look for relationship pattern: (src)-[rel]->(tgt)
        rel_pattern = re.search(r"\(([^)]+)\)\s*-\s*\[([^\]]*)\]\s*->\s*\(([^)]+)\)", create_body)
        if rel_pattern:
            src_str, rel_str, tgt_str = rel_pattern.groups()
            src_node = self._parse_and_create_node(src_str, node_var_map)
            tgt_node = self._parse_and_create_node(tgt_str, node_var_map)
            created_nodes.extend([src_node, tgt_node])
            
            # Parse relationship
            rel = self._parse_and_create_rel(rel_str, src_node.id, tgt_node.id)
            created_rels.append(rel)
        else:
            # Maybe multiple nodes comma separated: (n1:...), (n2:...)
            node_patterns = re.findall(r"\(([^)]+)\)", create_body)
            if not node_patterns:
                raise ValueError("Could not parse node pattern in CREATE clause. Example: CREATE (n:Student {name: 'John'})")
            for np in node_patterns:
                node = self._parse_and_create_node(np, node_var_map)
                created_nodes.append(node)

        # Build output dataframe
        rows = []
        for n in created_nodes:
            d = {"Entity": "Node", "ID": n.id, "Labels": ":".join(n.labels)}
            d.update(n.properties)
            rows.append(d)
        for r in created_rels:
            d = {"Entity": "Relationship", "ID": r.id, "Type": r.type, "Source": r.source, "Target": r.target}
            d.update(r.properties)
            rows.append(d)

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        msg = f"Created {len(created_nodes)} node(s) and {len(created_rels)} relationship(s)."

        return {
            "success": True,
            "message": msg,
            "dataframe": df,
            "matched_node_ids": [n.id for n in created_nodes],
            "matched_rel_ids": [r.id for r in created_rels],
            "affected_count": len(created_nodes) + len(created_rels)
        }

    def _parse_and_create_node(self, node_str: str, node_var_map: Dict[str, Node]) -> Node:
        # Syntax: var:Label1:Label2 {k: v, k2: v2}
        node_str = node_str.strip()
        var_name = ""
        labels = []
        props = {}

        # Extract props if any
        m_props = re.search(r"\{([^}]+)\}", node_str)
        if m_props:
            props = self._parse_props_string(m_props.group(1))
            prefix = node_str[:m_props.start()].strip()
        else:
            prefix = node_str

        parts = prefix.split(":")
        var_name = parts[0].strip()
        if len(parts) > 1:
            labels = [p.strip() for p in parts[1:] if p.strip()]

        # Generate unique ID if not given in properties
        node_id = props.get("id") or props.get("name") or var_name
        if not node_id:
            node_id = f"node_{len(self.graph.nodes) + 1}"
        node_id = str(node_id).replace(" ", "_").lower()

        # If var_name was already matched or exists in graph
        if node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]
            if labels:
                for l in labels:
                    node.labels.add(l)
                    self.graph.label_index.setdefault(l, set()).add(node_id)
            node.properties.update(props)
        else:
            node = self.graph.add_node(node_id, labels, props)

        if var_name:
            node_var_map[var_name] = node
        return node

    def _parse_and_create_rel(self, rel_str: str, src_id: str, tgt_id: str) -> Relationship:
        # Syntax: var:TYPE {props} or :TYPE {props}
        rel_str = rel_str.strip()
        props = {}
        m_props = re.search(r"\{([^}]+)\}", rel_str)
        if m_props:
            props = self._parse_props_string(m_props.group(1))
            prefix = rel_str[:m_props.start()].strip()
        else:
            prefix = rel_str

        rel_type = "RELATED_TO"
        if ":" in prefix:
            rel_type = prefix.split(":")[-1].strip().upper()

        return self.graph.add_relationship(src_id, tgt_id, rel_type, props)

    def _parse_props_string(self, props_str: str) -> Dict[str, Any]:
        """Parses {name: 'Alice', gpa: 9.1, active: true}."""
        res = {}
        # Split by comma outside quotes
        pairs = re.findall(r'(\w+)\s*:\s*([^\'"][^,]*|\'[^\']*\'|"[^"]*")', props_str)
        for k, v in pairs:
            k = k.strip()
            v = v.strip().strip("'\"")
            # Type casting
            if v.lower() == "true":
                res[k] = True
            elif v.lower() == "false":
                res[k] = False
            else:
                try:
                    if "." in v:
                        res[k] = float(v)
                    else:
                        res[k] = int(v)
                except ValueError:
                    res[k] = v
        return res

    # ----------------------------------------------------------------------------------
    # MATCH EXECUTION (Pattern matching, WHERE, RETURN, SET, DELETE, DETACH DELETE)
    # ----------------------------------------------------------------------------------
    def _execute_match(self, query: str) -> Dict[str, Any]:
        # Identify main clauses in MATCH query:
        # MATCH <patterns>
        # [WHERE <conditions>]
        # [SET <assignments>]
        # [DELETE / DETACH DELETE <vars>]
        # [RETURN <items>] [ORDER BY ...] [LIMIT ...]

        # Split query into tokens while respecting quotes
        clause_regex = r"\b(MATCH|WHERE|SET|DETACH\s+DELETE|DELETE|CREATE|RETURN)\b"
        tokens = re.split(clause_regex, query, flags=re.IGNORECASE)

        clauses = {}
        i = 1
        while i < len(tokens):
            c_name = tokens[i].strip().upper()
            c_val = tokens[i+1].strip() if i+1 < len(tokens) else ""
            clauses[c_name] = c_val
            i += 2

        match_clause = clauses.get("MATCH", "")
        where_clause = clauses.get("WHERE", "")
        set_clause = clauses.get("SET", "")
        detach_del_clause = clauses.get("DETACH DELETE", "")
        del_clause = clauses.get("DELETE", "")
        create_in_match = clauses.get("CREATE", "")
        return_clause = clauses.get("RETURN", "")

        # 1. Match patterns to generate candidate variable bindings
        bindings, path_nodes, path_rels = self._match_pattern(match_clause)

        # 2. Filter bindings using WHERE clause
        if where_clause:
            bindings = [b for b in bindings if self._evaluate_where(where_clause, b)]

        # If no bindings matched
        if not bindings:
            return {
                "success": True,
                "message": "Query executed successfully: (0 records returned / 0 matches found)",
                "dataframe": pd.DataFrame(),
                "matched_node_ids": [],
                "matched_rel_ids": [],
                "affected_count": 0
            }

        # 3. Check for mutation: SET
        if set_clause:
            affected = self._execute_set(set_clause, bindings)
            return {
                "success": True,
                "message": f"Updated {affected} properties across {len(bindings)} matched record(s).",
                "dataframe": pd.DataFrame([self._binding_to_row(b) for b in bindings]),
                "matched_node_ids": list(set(path_nodes)),
                "matched_rel_ids": list(set(path_rels)),
                "affected_count": affected
            }

        # 4. Check for mutation: DELETE or DETACH DELETE
        if detach_del_clause or del_clause:
            detach = bool(detach_del_clause)
            del_targets = (detach_del_clause or del_clause).split(",")
            del_targets = [t.strip() for t in del_targets]
            
            deleted_nodes = 0
            deleted_rels = 0
            errors = []

            for b in bindings:
                for target in del_targets:
                    obj = b.get(target)
                    if isinstance(obj, Node):
                        succ, msg = self.graph.delete_node(obj.id, detach=detach)
                        if succ:
                            deleted_nodes += 1
                        else:
                            errors.append(msg)
                    elif isinstance(obj, Relationship):
                        succ, msg = self.graph.delete_relationship(obj.id)
                        if succ:
                            deleted_rels += 1
                        else:
                            errors.append(msg)

            if errors:
                return {
                    "success": False,
                    "message": " | ".join(errors[:3]),
                    "dataframe": pd.DataFrame(),
                    "matched_node_ids": [],
                    "matched_rel_ids": [],
                    "affected_count": 0
                }

            return {
                "success": True,
                "message": f"Deleted {deleted_nodes} node(s) and {deleted_rels} relationship(s).",
                "dataframe": pd.DataFrame([{"Deleted_Nodes": deleted_nodes, "Deleted_Relationships": deleted_rels}]),
                "matched_node_ids": [],
                "matched_rel_ids": [],
                "affected_count": deleted_nodes + deleted_rels
            }

        # 5. Check for CREATE in MATCH: MATCH ... CREATE (a)-[r]->(b)
        if create_in_match:
            # We can create relationships between matched nodes
            affected_rels = []
            rel_pattern = re.search(r"\(([^)]+)\)\s*-\s*\[([^\]]*)\]\s*->\s*\(([^)]+)\)", create_in_match)
            if rel_pattern:
                src_var, rel_spec, tgt_var = rel_pattern.groups()
                src_var = src_var.strip()
                tgt_var = tgt_var.strip()
                for b in bindings:
                    src_obj = b.get(src_var)
                    tgt_obj = b.get(tgt_var)
                    if isinstance(src_obj, Node) and isinstance(tgt_obj, Node):
                        r = self._parse_and_create_rel(rel_spec, src_obj.id, tgt_obj.id)
                        affected_rels.append(r)

            return {
                "success": True,
                "message": f"Created {len(affected_rels)} relationship(s) between matched nodes.",
                "dataframe": pd.DataFrame([r.to_dict() for r in affected_rels]) if affected_rels else pd.DataFrame(),
                "matched_node_ids": list(set(path_nodes)),
                "matched_rel_ids": [r.id for r in affected_rels],
                "affected_count": len(affected_rels)
            }

        # 6. RETURN clause projection & Aggregation
        df, matched_nids, matched_rids = self._project_return(return_clause, bindings)

        return {
            "success": True,
            "message": f"Returned {len(df)} row(s).",
            "dataframe": df,
            "matched_node_ids": matched_nids,
            "matched_rel_ids": matched_rids,
            "affected_count": len(df)
        }

    # ----------------------------------------------------------------------------------
    # PATTERN MATCHING ALGORITHM
    # ----------------------------------------------------------------------------------
    def _match_pattern(self, match_str: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """
        Parses MATCH pattern string:
        Supports:
          - (n) or (n:Label) or (n:Label {k: 'v'})
          - (a)-[r:TYPE]->(b) or (a)<-[r:TYPE]-(b) or (a)-[r:TYPE]-(b)
          - (a)-[:TYPE*1..2]->(b) (variable-length traversal)
          - Multi-hop chains: (a)-[r1]->(b)-[r2]->(c)
          - Comma-separated disconnected patterns: (a:Student), (c:Course)
        """
        all_bindings = []
        matched_nids = set()
        matched_rids = set()

        # Check if chained pattern: (a)-[r1]->(b)-[r2]->(c) or (a)-[r1]->(b)<-[r2]-(c)
        chain_tokens = re.split(r"(->|<-|-)", match_str)
        
        # If it's a simple comma-separated list of nodes: MATCH (s:Student), (c:Course)
        if "," in match_str and not ("-" in match_str):
            sub_patterns = [p.strip() for p in match_str.split(",") if p.strip()]
            sub_bindings_list = []
            for sp in sub_patterns:
                b_list, _, _ = self._match_single_node_pattern(sp)
                sub_bindings_list.append(b_list)
            # Cartesian product
            import itertools
            combined = []
            for combo in itertools.product(*sub_bindings_list):
                merged = {}
                for item in combo:
                    merged.update(item)
                combined.append(merged)
            for b in combined:
                for val in b.values():
                    if isinstance(val, Node):
                        matched_nids.add(val.id)
            return combined, list(matched_nids), []

        # Parse path expressions
        # Find all nodes: (var:Label {props}) and relationships: [var:TYPE {props}]
        # Tokenize by relationships:
        # Example pattern: (a:Student)-[r:ENROLLED_IN]->(c:Course)
        # Regex for link: \(([^)]+)\)\s*([<-]{1,2}\[[^\]]*\][>-]{1,2})\s*\(([^)]+)\)
        link_matches = list(re.finditer(r"\(([^)]+)\)\s*([<-]{1,2}\[[^\]]*\][>-]{1,2})", match_str))

        if not link_matches:
            # Single node match: MATCH (n:Student)
            m_node = re.search(r"\(([^)]+)\)", match_str)
            if m_node:
                bindings, nids, _ = self._match_single_node_pattern(m_node.group(1))
                return bindings, nids, []
            else:
                raise ValueError(f"Unrecognized MATCH pattern: '{match_str}'")

        # Parse general multi-hop path: (n0) [e0] (n1) [e1] (n2) ...
        node_strs = re.findall(r"\(([^)]+)\)", match_str)
        edge_strs = re.findall(r"([<-]{1,2}\[[^\]]*\][>-]{1,2})", match_str)

        if len(node_strs) != len(edge_strs) + 1:
            raise ValueError(f"Malformed path in MATCH pattern: '{match_str}'")

        # Parse node constraints
        node_constraints = [self._parse_node_pattern_spec(ns) for ns in node_strs]
        edge_constraints = [self._parse_edge_pattern_spec(es) for es in edge_strs]

        # Traversal search starting from candidate nodes for node 0
        cand_start_nodes = self._find_candidate_nodes(node_constraints[0])

        for start_node in cand_start_nodes:
            initial_binding = {}
            if node_constraints[0]["var"]:
                initial_binding[node_constraints[0]["var"]] = start_node
            
            # Recursive path extension
            self._extend_path(
                current_node=start_node,
                step_idx=0,
                node_constraints=node_constraints,
                edge_constraints=edge_constraints,
                current_binding=initial_binding,
                all_bindings=all_bindings,
                matched_nids=matched_nids,
                matched_rids=matched_rids
            )

        return all_bindings, list(matched_nids), list(matched_rids)

    def _extend_path(self, current_node: Node, step_idx: int,
                     node_constraints: List[Dict], edge_constraints: List[Dict],
                     current_binding: Dict[str, Any], all_bindings: List[Dict[str, Any]],
                     matched_nids: Set[str], matched_rids: Set[str]):
        if step_idx >= len(edge_constraints):
            # Reached full path length!
            all_bindings.append(dict(current_binding))
            for v in current_binding.values():
                if isinstance(v, Node):
                    matched_nids.add(v.id)
                elif isinstance(v, Relationship):
                    matched_rids.add(v.id)
            return

        edge_c = edge_constraints[step_idx]
        next_node_c = node_constraints[step_idx + 1]

        # Determine direction:
        # -> means outgoing from current_node
        # <- means incoming to current_node
        # - means either direction
        direction = edge_c["direction"]
        candidate_rel_ids = []
        if direction == "outgoing":
            candidate_rel_ids = self.graph.outgoing.get(current_node.id, [])
        elif direction == "incoming":
            candidate_rel_ids = self.graph.incoming.get(current_node.id, [])
        else:  # undirected / bidirectional
            candidate_rel_ids = self.graph.outgoing.get(current_node.id, []) + self.graph.incoming.get(current_node.id, [])

        for rel_id in candidate_rel_ids:
            rel = self.graph.relationships.get(rel_id)
            if not rel:
                continue

            # Check rel type
            if edge_c["type"] and rel.type != edge_c["type"]:
                continue
            # Check rel properties
            if edge_c["props"] and not self._props_match(rel.properties, edge_c["props"]):
                continue

            # Determine next node in this hop
            if direction == "outgoing":
                next_nid = rel.target
            elif direction == "incoming":
                next_nid = rel.source
            else:
                next_nid = rel.target if rel.source == current_node.id else rel.source

            next_node = self.graph.nodes.get(next_nid)
            if not next_node:
                continue

            # Check next node constraint
            if not self._node_matches_constraint(next_node, next_node_c):
                continue

            # Check if next_node variable was already bound
            if next_node_c["var"] and next_node_c["var"] in current_binding:
                if current_binding[next_node_c["var"]].id != next_node.id:
                    continue

            # Add to binding and proceed
            new_binding = dict(current_binding)
            if edge_c["var"]:
                new_binding[edge_c["var"]] = rel
            if next_node_c["var"]:
                new_binding[next_node_c["var"]] = next_node

            self._extend_path(
                current_node=next_node,
                step_idx=step_idx + 1,
                node_constraints=node_constraints,
                edge_constraints=edge_constraints,
                current_binding=new_binding,
                all_bindings=all_bindings,
                matched_nids=matched_nids,
                matched_rids=matched_rids
            )

    def _match_single_node_pattern(self, node_str: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        spec = self._parse_node_pattern_spec(node_str)
        cand_nodes = self._find_candidate_nodes(spec)
        bindings = []
        nids = []
        for n in cand_nodes:
            b = {}
            if spec["var"]:
                b[spec["var"]] = n
            bindings.append(b)
            nids.append(n.id)
        return bindings, nids, []

    def _parse_node_pattern_spec(self, node_str: str) -> Dict[str, Any]:
        node_str = node_str.strip()
        props = {}
        m_props = re.search(r"\{([^}]+)\}", node_str)
        if m_props:
            props = self._parse_props_string(m_props.group(1))
            prefix = node_str[:m_props.start()].strip()
        else:
            prefix = node_str

        parts = prefix.split(":")
        var_name = parts[0].strip() if parts[0].strip() else None
        labels = [p.strip() for p in parts[1:] if p.strip()]
        return {"var": var_name, "labels": labels, "props": props}

    def _parse_edge_pattern_spec(self, edge_raw: str) -> Dict[str, Any]:
        edge_raw = edge_raw.strip()
        direction = "undirected"
        if edge_raw.startswith("<-") and edge_raw.endswith("-"):
            direction = "incoming"
        elif edge_raw.startswith("-") and edge_raw.endswith("->"):
            direction = "outgoing"
        elif edge_raw.startswith("<-") and edge_raw.endswith("->"):
            direction = "undirected"
        
        # Extract inside brackets [ ... ]
        m_bracket = re.search(r"\[([^\]]*)\]", edge_raw)
        inside = m_bracket.group(1).strip() if m_bracket else ""

        props = {}
        m_props = re.search(r"\{([^}]+)\}", inside)
        if m_props:
            props = self._parse_props_string(m_props.group(1))
            inside = inside[:m_props.start()].strip()

        # Variable length check: [:REL*1..2]
        var_len = None
        if "*" in inside:
            parts_star = inside.split("*")
            inside = parts_star[0].strip()
            var_len = parts_star[1].strip()

        parts = inside.split(":")
        var_name = parts[0].strip() if parts[0].strip() else None
        rel_type = parts[1].strip().upper() if len(parts) > 1 and parts[1].strip() else None

        return {
            "direction": direction,
            "var": var_name,
            "type": rel_type,
            "props": props,
            "var_len": var_len
        }

    def _find_candidate_nodes(self, spec: Dict[str, Any]) -> List[Node]:
        labels = spec["labels"]
        props = spec["props"]

        if labels:
            # Intersect label indices
            cand_ids = None
            for lbl in labels:
                s = self.graph.label_index.get(lbl, set())
                cand_ids = s if cand_ids is None else cand_ids.intersection(s)
            cand_nodes = [self.graph.nodes[nid] for nid in cand_ids]
        else:
            cand_nodes = list(self.graph.nodes.values())

        if props:
            cand_nodes = [n for n in cand_nodes if self._props_match(n.properties, props)]

        return cand_nodes

    def _node_matches_constraint(self, node: Node, spec: Dict[str, Any]) -> bool:
        if spec["labels"]:
            for lbl in spec["labels"]:
                if lbl not in node.labels:
                    return False
        if spec["props"]:
            if not self._props_match(node.properties, spec["props"]):
                return False
        return True

    def _props_match(self, entity_props: Dict[str, Any], query_props: Dict[str, Any]) -> bool:
        for k, v in query_props.items():
            if str(entity_props.get(k, "")).lower() != str(v).lower():
                return False
        return True

    # ----------------------------------------------------------------------------------
    # WHERE EVALUATION
    # ----------------------------------------------------------------------------------
    def _evaluate_where(self, where_str: str, binding: Dict[str, Any]) -> bool:
        """
        Evaluates WHERE clauses like:
          s.gpa >= 8.5 AND c.credits > 3
          s.dept = 'CSE' OR s.name CONTAINS 'Alice'
        """
        # Support basic AND / OR chaining
        or_clauses = re.split(r"\bOR\b", where_str, flags=re.IGNORECASE)
        for or_c in or_clauses:
            and_clauses = re.split(r"\bAND\b", or_c, flags=re.IGNORECASE)
            and_result = True
            for cond in and_clauses:
                if not self._evaluate_single_condition(cond.strip(), binding):
                    and_result = False
                    break
            if and_result:
                return True
        return False

    def _evaluate_single_condition(self, cond: str, binding: Dict[str, Any]) -> bool:
        # Operators: CONTAINS, STARTS WITH, ENDS WITH, >=, <=, <>, !=, =, >, <
        op_patterns = [
            (r"\bCONTAINS\b", lambda a, b: str(b).lower() in str(a).lower()),
            (r"\bSTARTS\s+WITH\b", lambda a, b: str(a).lower().startswith(str(b).lower())),
            (r"\bENDS\s+WITH\b", lambda a, b: str(a).lower().endswith(str(b).lower())),
            (r">=", lambda a, b: float(a) >= float(b)),
            (r"<=", lambda a, b: float(a) <= float(b)),
            (r"<>|!=", lambda a, b: str(a) != str(b)),
            (r"=", lambda a, b: str(a).lower() == str(b).lower()),
            (r">", lambda a, b: float(a) > float(b)),
            (r"<", lambda a, b: float(a) < float(b))
        ]

        for pattern, func in op_patterns:
            m = re.search(pattern, cond, flags=re.IGNORECASE)
            if m:
                left_expr = cond[:m.start()].strip()
                right_expr = cond[m.end():].strip()
                left_val = self._resolve_expr_value(left_expr, binding)
                right_val = self._resolve_expr_value(right_expr, binding)
                if left_val is None or right_val is None:
                    return False
                try:
                    return func(left_val, right_val)
                except (ValueError, TypeError):
                    return False

        return True

    def _resolve_expr_value(self, expr: str, binding: Dict[str, Any]) -> Any:
        expr = expr.strip()
        # Check if literal string: 'Alice' or "Alice"
        if (expr.startswith("'") and expr.endswith("'")) or (expr.startswith('"') and expr.endswith('"')):
            return expr[1:-1]
        # Check if number
        try:
            if "." in expr:
                return float(expr)
            return int(expr)
        except ValueError:
            pass
        # Check if property reference: var.prop
        if "." in expr:
            var, prop = expr.split(".", 1)
            obj = binding.get(var.strip())
            if obj:
                return obj.get(prop.strip())
            return None
        # Check if bare variable (returns id or display name)
        if expr in binding:
            obj = binding[expr]
            if isinstance(obj, Node):
                return obj.id
            if isinstance(obj, Relationship):
                return obj.type
        return expr

    # ----------------------------------------------------------------------------------
    # SET EXECUTION
    # ----------------------------------------------------------------------------------
    def _execute_set(self, set_clause: str, bindings: List[Dict[str, Any]]) -> int:
        """
        Executes SET clause:
          SET s.gpa = 9.2, s.status = 'Graduated'
        """
        assignments = [a.strip() for a in set_clause.split(",") if a.strip()]
        affected_count = 0

        for b in bindings:
            for assign in assignments:
                if "=" in assign:
                    target, val_expr = assign.split("=", 1)
                    target = target.strip()
                    val = self._resolve_expr_value(val_expr.strip(), b)
                    if "." in target:
                        var, prop = target.split(".", 1)
                        obj = b.get(var.strip())
                        if isinstance(obj, Node):
                            obj.properties[prop.strip()] = val
                            affected_count += 1
                        elif isinstance(obj, Relationship):
                            obj.properties[prop.strip()] = val
                            affected_count += 1
        return affected_count

    # ----------------------------------------------------------------------------------
    # RETURN PROJECTION & AGGREGATION
    # ----------------------------------------------------------------------------------
    def _project_return(self, return_clause: str, bindings: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str], List[str]]:
        matched_nids = []
        matched_rids = []

        for b in bindings:
            for v in b.values():
                if isinstance(v, Node):
                    matched_nids.append(v.id)
                elif isinstance(v, Relationship):
                    matched_rids.append(v.id)

        matched_nids = list(set(matched_nids))
        matched_rids = list(set(matched_rids))

        if not return_clause or return_clause.strip() == "*":
            # Return all variables
            rows = [self._binding_to_row(b) for b in bindings]
            return pd.DataFrame(rows), matched_nids, matched_rids

        # Parse comma-separated projections: s.name AS student_name, count(c) AS total_courses
        items = [item.strip() for item in return_clause.split(",") if item.strip()]

        # Check for aggregation functions: count(...), avg(...), sum(...), min(...), max(...), collect(...)
        has_aggregation = any(re.search(r"\b(COUNT|AVG|SUM|MIN|MAX|COLLECT)\s*\(", item, flags=re.IGNORECASE) for item in items)

        if has_aggregation:
            return self._aggregate_return(items, bindings), matched_nids, matched_rids

        # Non-aggregated projection
        rows = []
        for b in bindings:
            row = {}
            for item in items:
                col_name, col_val = self._eval_return_item(item, b)
                row[col_name] = col_val
            rows.append(row)

        df = pd.DataFrame(rows)
        return df, matched_nids, matched_rids

    def _eval_return_item(self, item_str: str, binding: Dict[str, Any]) -> Tuple[str, Any]:
        alias = None
        m_alias = re.search(r"\bAS\b\s+(\w+)$", item_str, flags=re.IGNORECASE)
        if m_alias:
            alias = m_alias.group(1).strip()
            expr = item_str[:m_alias.start()].strip()
        else:
            expr = item_str.strip()

        col_name = alias if alias else expr

        if "." in expr:
            var, prop = expr.split(".", 1)
            obj = binding.get(var.strip())
            if obj:
                return col_name, obj.get(prop.strip(), None)
            return col_name, None

        if expr in binding:
            obj = binding[expr]
            if isinstance(obj, Node):
                return col_name, f"(:{':'.join(obj.labels)} {{name: '{obj.display_name()}'}})"
            if isinstance(obj, Relationship):
                return col_name, f"[:{obj.type} {obj.properties}]"
            return col_name, str(obj)

        return col_name, expr

    def _aggregate_return(self, items: List[str], bindings: List[Dict[str, Any]]) -> pd.DataFrame:
        # Separate group keys from aggregate functions
        group_keys = []
        agg_specs = []

        for item in items:
            alias = None
            m_alias = re.search(r"\bAS\b\s+(\w+)$", item, flags=re.IGNORECASE)
            if m_alias:
                alias = m_alias.group(1).strip()
                expr = item[:m_alias.start()].strip()
            else:
                expr = item.strip()
            col_name = alias if alias else expr

            m_agg = re.match(r"(COUNT|AVG|SUM|MIN|MAX|COLLECT)\s*\((.*?)\)", expr, flags=re.IGNORECASE)
            if m_agg:
                fn_name = m_agg.group(1).upper()
                fn_arg = m_agg.group(2).strip()
                agg_specs.append((col_name, fn_name, fn_arg))
            else:
                group_keys.append((col_name, expr))

        # Group bindings
        groups = {}
        for b in bindings:
            # Determine group key
            key_vals = tuple(self._resolve_expr_value(expr, b) for _, expr in group_keys)
            groups.setdefault(key_vals, []).append(b)

        result_rows = []
        for key_vals, group_bindings in groups.items():
            row = {}
            for idx, (col_name, _) in enumerate(group_keys):
                row[col_name] = key_vals[idx]

            # Compute aggregates
            for col_name, fn_name, fn_arg in agg_specs:
                if fn_name == "COUNT":
                    if fn_arg == "*" or not fn_arg:
                        row[col_name] = len(group_bindings)
                    else:
                        vals = [self._resolve_expr_value(fn_arg, b) for b in group_bindings]
                        vals = [v for v in vals if v is not None]
                        row[col_name] = len(vals)
                elif fn_name in ("AVG", "SUM", "MIN", "MAX"):
                    vals = [self._resolve_expr_value(fn_arg, b) for b in group_bindings]
                    nums = []
                    for v in vals:
                        try:
                            nums.append(float(v))
                        except (ValueError, TypeError):
                            pass
                    if not nums:
                        row[col_name] = None
                    elif fn_name == "AVG":
                        row[col_name] = round(sum(nums) / len(nums), 2)
                    elif fn_name == "SUM":
                        row[col_name] = round(sum(nums), 2)
                    elif fn_name == "MIN":
                        row[col_name] = min(nums)
                    elif fn_name == "MAX":
                        row[col_name] = max(nums)
                elif fn_name == "COLLECT":
                    vals = [self._resolve_expr_value(fn_arg, b) for b in group_bindings]
                    row[col_name] = vals

            result_rows.append(row)

        return pd.DataFrame(result_rows)

    def _binding_to_row(self, binding: Dict[str, Any]) -> Dict[str, Any]:
        row = {}
        for k, v in binding.items():
            if isinstance(v, Node):
                row[f"{k}.id"] = v.id
                row[f"{k}.labels"] = ":".join(v.labels)
                for pk, pv in v.properties.items():
                    row[f"{k}.{pk}"] = pv
            elif isinstance(v, Relationship):
                row[f"{k}.type"] = v.type
                row[f"{k}.source"] = v.source
                row[f"{k}.target"] = v.target
                for pk, pv in v.properties.items():
                    row[f"{k}.{pk}"] = pv
            else:
                row[k] = str(v)
        return row
