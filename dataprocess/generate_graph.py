import numpy as np
import networkx as nx
import scipy.sparse as sps

def refill_e(edges, n, amount):
    if amount == 0:
        return edges
    # print(edges)
    # ee = np.sort(edges).tolist()
    ee = {tuple(row) for row in np.sort(edges).tolist()}
    new_e = []
    check = 0
    while len(new_e) < amount:
        _e = np.random.randint(n, size=2)
        # _ee = np.sort(_e).tolist()
        _ee = tuple(np.sort(_e).tolist())
        check += 1
        if not(_ee in ee) and _e[0] != _e[1]:
            # ee.append(_ee)
            ee.add(_ee)
            new_e.append(_e)
            check = 0
            # print(f"refill - {len(new_e)}/{amount}")
        if check % 1000 == 999:
            print(f"refill - {check + 1} times in a row fail")
    # print(new_e)
    return np.append(edges, new_e, axis=0)


@ ex.capture
def remove_e(edges, noise, no_disc=True, until_connected=False):
    ii = 0
    while True:
        ii += 1
        print(f"##<{ii}>##")

        if no_disc:
            bin_count = np.bincount(edges.flatten())
            rows_to_delete = []
            for i, edge in enumerate(edges):
                if np.random.sample(1)[0] < noise:
                    e, f = edge
                    if bin_count[e] > 1 and bin_count[f] > 1:
                        bin_count[e] -= 1
                        bin_count[f] -= 1
                        rows_to_delete.append(i)
            new_edges = np.delete(edges, rows_to_delete, axis=0)
        else:
            new_edges = edges[np.random.sample(edges.shape[0]) >= noise]

        graph = nx.Graph(new_edges.tolist())
        graph_cc = len(max(nx.connected_components(graph), key=len))
        print(graph_cc, np.amax(edges)+1)
        graph_connected = graph_cc == np.amax(edges) + 1
        # if not graph_connected:
        #     break
        if graph_connected or not until_connected:
            break
    return new_edges


def load_as_nx(path):
    G_e = np.loadtxt(path, int)
    G = nx.Graph(G_e.tolist())
    print("Just checking",nx.is_directed(G))
    return np.array(G.edges)


def loadnx(path):
    G_e = np.loadtxt(path, int)
    return nx.Graph(G_e.tolist())


@ ex.capture
def noise_types(noise_level, noise_type=1):
    return [
        {'target_noise': noise_level},
        {'target_noise': noise_level, 'refill': True},
        {'source_noise': noise_level, 'target_noise': noise_level},
    ][noise_type - 1]


def generate_graphs(G, source_noise=0.00, target_noise=0.00, refill=False):

    if isinstance(G, list):

        _src, _tar, _gt = G

        Src_e = load_as_nx(_src)
        Tar_e = load_as_nx(_tar)

        if isinstance(_gt, str):
            gt_e = np.loadtxt(_gt, int).T
        elif _gt is None:
            gt1 = np.arange(
                max(np.amax(Src_e), np.amax(Tar_e))+1).reshape(-1, 1)
            gt_e = np.repeat(gt1, 2, axis=1).T
            print(gt_e)
        else:
            gt_e = np.array(_gt, int)

        Gt = (
            gt_e[:, gt_e[1].argsort()][0],
            gt_e[:, gt_e[0].argsort()][1]
        )

        return Src_e, Tar_e, Gt

        # dataset = G['dataset']
        # edges = G['edges']
        # noise_level = G['noise_level']

        # source = f"data/{dataset}/source.txt"
        # target = f"data/{dataset}/noise_level_{noise_level}/edges_{edges}.txt"
        # grand_truth = f"data/{dataset}/noise_level_{noise_level}/gt_{edges}.txt"

        # Src_e = load_as_nx(source)
        # Tar_e = load_as_nx(target)
        # gt_e = np.loadtxt(grand_truth, int).T

        # # Src = e_to_G(Src_e)
        # # Tar = e_to_G(Tar_e)

        # Gt = (
        #     gt_e[:, gt_e[1].argsort()][0],
        #     gt_e[:, gt_e[0].argsort()][1]
        # )

        # return Src_e, Tar_e, Gt
    elif isinstance(G, str):
        Src_e = load_as_nx(G)
    elif isinstance(G, nx.Graph):
        Src_e = np.array(G.edges)
    else:
        return sps.csr_matrix([]), sps.csr_matrix([]), (np.empty(1), np.empty(1))
    if (np.amin(Src_e)!=0):
        Src_e=Src_e-np.amin(Src_e)
    n = np.amax(Src_e) + 1
    nedges = Src_e.shape[0]

    gt_e = np.array((
        np.arange(n),
        np.random.permutation(n)
    ))

    Gt = (
        gt_e[:, gt_e[1].argsort()][0],
        gt_e[:, gt_e[0].argsort()][1]
    )

    Tar_e = Gt[0][Src_e]

    Src_e = remove_e(Src_e, source_noise)
    Tar_e = remove_e(Tar_e, target_noise)

    if refill:
        Src_e = refill_e(Src_e, n, nedges - Src_e.shape[0])
        Tar_e = refill_e(Tar_e, n, nedges - Tar_e.shape[0])

    return Src_e, Tar_e,  Gt


# def init1(graphs, iters, _run, path=None):
@ ex.capture
def init1(graphs, iters):

    # if path is None:
    #     S_G = np.memmap(f"runs/{_run._id}/_S_G.dat", dtype=object,
    #                     mode='w+', shape=(len(graphs), iters))

    #     for i, (alg, args) in enumerate(graphs):
    #         for j in range(iters):
    #             print(i, j)
    #             S_G[i, j] = alg(*args)
    #             S_G.flush()

    #     randcheck = np.random.rand(1)[0]

    # else:
    #     S_G = np.memmap(f"{path}/_S_G.dat", dtype=object,
    #                     mode='r', shape=(len(graphs), iters))

    #     randcheck = path

    S_G = [
        [alg(*args) for _ in range(iters)] for alg, args in graphs
    ]

    return S_G, np.random.rand(1)[0]


# def init2(S_G, noises, _run, path=None):
@ ex.capture
def init2(S_G, noises):

    # if path is None:
    #     G = np.memmap(f"runs/{_run._id}/_G.dat", dtype=object,
    #                   mode='w+', shape=(S_G.shape[0], len(noises), S_G.shape[1]))

    #     for i, gi in enumerate(S_G):
    #         for j, noise in enumerate(noises):
    #             for k, g in enumerate(gi):
    #                 G[i, j, k] = generate_graphs(g, **noise_types(noise))
    #                 G.flush()

    #     randcheck = np.random.rand(1)[0]

    # else:
    #     G = np.memmap(f"{path}/_S_G.dat", dtype=object,
    #                   mode='r', shape=(S_G.shape[0], len(noises), S_G.shape[1]))

    #     randcheck = path

    G = [
        [
            [
                generate_graphs(g, **noise_types(noise)) for g in gi
            ] for noise in noises
        ] for gi in S_G
    ]

    return G, np.random.rand(1)[0]