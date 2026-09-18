import os
import re

app_path = "C:/Users/HP/kgirs_CA/app.py"

with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add imports
if "import streamlit.components.v1 as components" not in content:
    content = content.replace("import streamlit as st", "import streamlit as st\nimport streamlit.components.v1 as components\nimport json")

# 2. Add vis-network rendering function before render_theory_section
vis_network_func = '''
def render_interactive_graph_canvas(graph: PropertyGraph, 
                                    matched_node_ids: Optional[List[str]] = None,
                                    matched_rel_ids: Optional[List[str]] = None) -> None:
    """Renders a Neo4j Bloom style interactive graph using vis-network."""
    
    nodes_data = []
    edges_data = []
    
    matched_nids = set(matched_node_ids) if matched_node_ids else set()
    matched_rids = set(matched_rel_ids) if matched_rel_ids else set()
    
    for nid, node in graph.nodes.items():
        primary_label = sorted(list(node.labels))[0] if node.labels else "Entity"
        base_color = LABEL_COLORS.get(primary_label, DEFAULT_NODE_COLOR)
        
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
            "label": f"<b>{node.display_name()}</b>\\n<i>:{primary_label}</i>",
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
                height: 600px;
                border: 1px solid #E2E8F0;
                background-color: #F8FAFC;
                border-radius: 8px;
            }}
            .vis-tooltip {{
                background-color: white !important;
                border: 1px solid #CBD5E1 !important;
                border-radius: 6px !important;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
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
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border: 1px solid #E2E8F0;
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
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                border: 1px solid #E2E8F0;
                max-width: 180px;
                font-size: 12px;
                max-height: 560px;
                overflow-y: auto;
            }}
            .legend-title {{ font-weight: 600; margin-bottom: 8px; color: #1E293B; border-bottom: 1px solid #E2E8F0; padding-bottom: 4px; }}
            .legend-item {{ display: flex; align-items: center; margin-bottom: 6px; justify-content: space-between; }}
            .legend-label-group {{ display: flex; align-items: center; gap: 6px; }}
            .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; border: 1px solid rgba(0,0,0,0.1); }}
            .badge {{ background: #F1F5F9; color: #64748B; padding: 2px 6px; border-radius: 12px; font-size: 10px; font-weight: 600; }}
            .rel-badge {{ background: #F1F5F9; border: 1px solid #CBD5E1; color: #475569; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
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
                <button class="ctrl-btn" onclick="network.fit({{animation: true}})" title="Fit to View">🎯 Fit</button>
                <button class="ctrl-btn" onclick="zoom(0.2)" title="Zoom In">➕</button>
                <button class="ctrl-btn" onclick="zoom(-0.2)" title="Zoom Out">➖</button>
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
    
    components.html(html_code, height=620)

# ======================================================================================
# 5. SECTION RENDERERS
'''

if "def render_interactive_graph_canvas" not in content:
    content = content.replace("# ======================================================================================\n# 5. SECTION RENDERERS", vis_network_func)


# 3. Update the UI section in render_simulation_section to use the new canvas!
old_viz_section = '''    # ----------------------------------------------------------------------------------
    # GRAPH VISUALIZATION & VIEW CONTROLS
    # ----------------------------------------------------------------------------------
    st.subheader("Real-Time Graph Visualization")

    col_v1, col_v2, col_v3 = st.columns([2, 2, 2])
    with col_v1:
        layout_opt = st.selectbox(
            "Graph Layout Algorithm:",
            options=["Spring (Force-Directed)", "Circular", "Kamada-Kawai", "Shell"],
            index=0
        )
    with col_v2:
        label_opt = st.selectbox(
            "Node Display Labels:",
            options=["Name / Label", "Name Only", "Node ID", "Label Only"],
            index=0
        )
    with col_v3:
        st.write("")
        st.write("")
        if st.button("Clear Highlighting", use_container_width=True):
            st.session_state["matched_node_ids"] = []
            st.session_state["matched_rel_ids"] = []
            st.rerun()

    fig = render_graph_figure(
        graph=graph,
        matched_node_ids=st.session_state.get("matched_node_ids"),
        matched_rel_ids=st.session_state.get("matched_rel_ids"),
        layout_algorithm=layout_opt,
        node_label_mode=label_opt
    )
    st.plotly_chart(fig, use_container_width=True)'''


new_viz_section = '''    # ----------------------------------------------------------------------------------
    # GRAPH VISUALIZATION & VIEW CONTROLS
    # ----------------------------------------------------------------------------------
    st.subheader("Interactive Graph Visualization")
    
    view_col, _, highlight_col = st.columns([3, 2, 1.5])
    with view_col:
        view_mode = st.radio("Graph Rendering Engine:", ["Interactive Neo4j Canvas (Recommended)", "Static Plotly Layout"], horizontal=True, label_visibility="collapsed")
    with highlight_col:
        if st.button("✨ Clear Highlighting", use_container_width=True):
            st.session_state["matched_node_ids"] = []
            st.session_state["matched_rel_ids"] = []
            st.rerun()

    if view_mode == "Interactive Neo4j Canvas (Recommended)":
        render_interactive_graph_canvas(
            graph=graph,
            matched_node_ids=st.session_state.get("matched_node_ids"),
            matched_rel_ids=st.session_state.get("matched_rel_ids")
        )
    else:
        # Fallback to Plotly with options
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
        st.plotly_chart(fig, use_container_width=True)'''

content = content.replace(old_viz_section, new_viz_section)


# 4. Improve Theory, Quiz, Report Layout spacing and typography.
# Theory Section refinements:
content = content.replace('st.header("Theoretical Framework: Graph Databases & Neo4j")', 'st.header("📖 Theoretical Framework: Graph Databases & Neo4j")\\n    st.markdown("---")')
content = content.replace('st.header("Interactive Graph Database Sandbox (Simulation)")', 'st.header("⚙️ Interactive Graph Database Sandbox")\\n    st.markdown("---")')
content = content.replace('st.header("Concept Assessment Quiz")', 'st.header("📝 Concept Assessment Quiz")\\n    st.markdown("---")')
content = content.replace('st.header("Report Generation")', 'st.header("📄 Official Report Generation")\\n    st.markdown("---")')

# Adjust metrics spacing
content = content.replace('with st.container(border=True):\\n        m1, m2, m3, m4, m5 = st.columns(5)', 'with st.container(border=True):\\n        st.markdown("**Graph Status Overview**")\\n        m1, m2, m3, m4, m5 = st.columns(5)')

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated app.py with Interactive Graph Visualization and improved UI.")
