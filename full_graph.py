import pydot

from chopsticks.env.state import ChopsticksState, Turn

if __name__ == "__main__":
    graph = pydot.Dot(
        "Decision Tree for Chopsticks", graph_type="graph", bgcolor="lightgray"
    )

    for l1 in range(5):
        for r1 in range(5):
            for l2 in range(5):
                for r2 in range(5):
                    for turn in [Turn.P1, Turn.P2]:
                        state = ChopsticksState(l1, r1, l2, r2, turn)
                        shape = "house" if turn == Turn.P1 else "invhouse"
                        if state.is_terminal():
                            color = (
                                "darkgreen" if state.winner() == Turn.P1 else "crimson"
                            )
                            node = pydot.Node(
                                str(state),
                                label=str(state),
                                color=color,
                                shape=shape,
                            )
                        else:
                            node = pydot.Node(
                                str(state),
                                label=str(state),
                                color="gray39",
                                shape=shape,
                            )
                        graph.add_node(node)

    for l1 in range(5):
        for r1 in range(5):
            for l2 in range(5):
                for r2 in range(5):
                    for turn in [Turn.P1, Turn.P2]:
                        state = ChopsticksState(l1, r1, l2, r2, turn)
                        for action in state.legal_moves():
                            child = state.transition(action)
                            edge = pydot.Edge(
                                str(state),
                                str(child),
                                label=str(action),
                                arrowhead="normal",
                                dir="forward",
                            )
                            graph.add_edge(edge)

    graph.write("full_graph.dot", format="raw")
    # graph.write("full_graph.png", format="png")
