import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
import functools
import pdb


class MLP(torch.nn.Module):
    def __init__(self, *sizes, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.2))
                if dropout:
                    fcs.append(torch.nn.Dropout(p=0.2))
                if batchnorm:
                    fcs.append(torch.nn.BatchNorm1d(sizes[i]))
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)


class NetConv(torch.nn.Module):
    """This Net Embedding Module corresponds to the Figure 2.a in the paper *https://guozz.cn/publication/mltimerdac-22/mltimerdac-22.pdf*
    models the wire delay.

    There are two main operators: (1) graph broadcast and (2) graph reduction.

    Graph Broadcast
    ----------------
    The information of "Net Driver + Net Sinks + Net Edge" are concatinated, pass through a MLP to obtain new features.
    CONCATINATE (Net Driver + Net Sinks + Net Edge)   -> MLP  ->  New features for "Net Sinks".


    Graph Reduce
    ----------------
    ....
    """

    def __init__(self, in_nf, in_ef, out_nf, h1=32, h2=32):
        super().__init__()
        self.in_nf = in_nf
        self.in_ef = in_ef
        self.out_nf = out_nf

        self.h1 = h1
        self.h2 = h2

        self.Message_MLP = MLP(
            self.in_nf * 2 + self.in_ef, 64, 64, 64, 1 + self.h1 + self.h2
        )
        self.Reduce_MLP = MLP(self.in_nf + self.h1 + self.h2, 64, 64, 64, self.out_nf)
        self.Broadcast_MLP = MLP(
            self.in_nf * 2 + self.in_ef, 64, 64, 64, 64, self.out_nf
        )

    def graph_broadcast(self, edges):
        """This `message_func` is used for generating new messages (efi). In  other words, new edge features"""
        net_driver_features = edges.src["nf"]  # X.f
        net_sink_features = edges.dst["nf"]  # a.f
        current_edge_features = edges.data["ef"]  # Distance (X->a)
        x = self.Broadcast_MLP(
            torch.cat(
                [net_driver_features, net_sink_features, current_edge_features], dim=1
            )
        )

        return {"efi": x}

    def graph_reduce(self, edges):
        x = torch.cat([edges.src["nf"], edges.dst["nf"], edges.data["ef"]], dim=1)
        x = self.Message_MLP(x)
        k, f1, f2 = torch.split(x, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)
        return {"efo1": f1 * k, "efo2": f2 * k}

    def graph_reduce_mlp_op(self, nodes):
        sum_features = nodes.data["nfo1"]
        max_features = nodes.data["nfo2"]
        driver_features = nodes.data["nf"]  # X.f
        x = torch.cat([driver_features, sum_features, max_features], dim=1)
        x = self.Reduce_MLP(x)
        return {
            "new_nf": x
        }  # this one corresponds to the `X's new feature` block in Figure 2.a

    def forward(self, g, ts, nf):
        with g.local_scope():

            # obtain NODE features

            # ======================================
            # Features
            # - is primary I/O pin or not (size: 1)
            # - is fanin or fanout (size: 1)
            # - distance to the 4 die area boundaries (size: 4)
            # - pin capacitance 4 (EL/RF)
            # ----------------------------------------------------
            # total number of features: 10

            # Tasks
            # - net delay to root pin    (size: 4 (EL/RF))
            # - arrival time             (size: 4 (EL/RF))
            # - slew                     (size: 4 (EL/RF))
            # - is timing endpoint or not     (size: 1)
            # - required arrival time for endpoints  (size: 4 (EL/RF))
            # =========================================

            # g.ndata['nf'] contains node features: size (#nodes, 10)
            g.ndata["nf"] = nf

            # [[UPDATE input nodes]]
            # the following func `g.update_all` will create new input nodes' features (new_nf).
            # https://docs.dgl.ai/en/1.1.x/generated/dgl.DGLGraph.update_all.html
            # (1) `self.graph_broadcast` will create a new feature named `efi` for all edge type `net_out` (edges that go out from output pins towards input pins).
            # (2) these newly create edge features `efi` will be aggerated by fn.sum() to create the new node feature named `new_nf`. IMPORTANTLY, on the input nodes (input pins) get this feature.
            # `new_nf` corresponds to the `a's new feature` block in Figure 2.a

            g.update_all(self.graph_broadcast, fn.sum("efi", "new_nf"), etype="net_out")

            # after this step, we get a.M, b.M, c.M ....

            # DGLGraph.apply_edges(func, edges='__ALL__', etype=None, inplace=False)¶
            # func: The function to generate new edge features.
            # etype: The type name of the edges.
            # After this step, each `net_in ` will have two new features: {"efo1": f1 * k, "efo2": f2 * k}
            g.apply_edges(self.graph_reduce, etype="net_in")

            # The model learn statistics from all net sinks through two reduction channels backed by "sum" and "max" operations.
            g.update_all(
                fn.copy_e("efo1", "efo1"), fn.sum("efo1", "nfo1"), etype="net_in"
            )
            g.update_all(
                fn.copy_e("efo2", "efo2"), fn.max("efo2", "nfo2"), etype="net_in"
            )

            # [[UPDATE output nodes]]
            # The statistics from all net sinks will be used to update the "driver" nodes. (iow: output nodes)
            g.apply_nodes(self.graph_reduce_mlp_op, ts["output_nodes"])
            return g.ndata["new_nf"]


class SignalProp(torch.nn.Module):
    """This module models the **cell delay interpolation** and the level-by-level arrival time and slew computation.
    The node embedding is propagated through each layer. How?: by alternating between net propagation layers and cell propagation layers.

    - net propagation + cell propagation layers are simular to graph broadcast and graph reduction  in the net embedding model, respectively.
    - howerver, cell propagation is a bit more complex as it involves cell delay computation with a cell library.


    [[Cell Delay Computation]]
    - In standard STA engines: cell_delay = F(driver_cell_type, net_load_statistic)
    - In TimingGCN:
    """

    def __init__(
        self,
        in_nf,
        in_cell_num_luts,
        in_cell_lut_sz,
        out_nf,
        out_cef,
        h1=32,
        h2=32,
        lut_dup=4,
    ):
        super().__init__()
        self.in_nf = in_nf
        self.in_cell_num_luts = in_cell_num_luts
        self.in_cell_lut_sz = in_cell_lut_sz
        self.out_nf = out_nf
        self.out_cef = out_cef
        self.h1 = h1
        self.h2 = h2
        self.lut_dup = lut_dup

        # https://i.imgur.com/SIBvvLK.png
        self.Net_Propogation_Layer_MLP = MLP(
            self.out_nf + 2 * self.in_nf, 64, 64, 64, 64, self.out_nf
        )

        # "Cell Propagation Layer" Block
        # =====================================================================
        self.Cell_Propagation_Layer_NLDM_Query_MLP = MLP(
            self.out_nf + 2 * self.in_nf,
            64,
            64,
            64,
            self.in_cell_num_luts * lut_dup * 2,
        )
        self.Cell_Propagation_Layer_LUT_Mask_MLP = MLP(
            1 + 2 + self.in_cell_lut_sz * 2, 64, 64, 64, self.in_cell_lut_sz * 2
        )
        self.Cell_Propagation_Layer_Cellprop_MLP = MLP(
            self.out_nf + 2 * self.in_nf + self.in_cell_num_luts * self.lut_dup,
            64,
            64,
            64,
            1 + self.h1 + self.h2 + self.out_cef,
        )
        self.Cell_Propagation_Layer_Cellprop_Reduce_MLP = MLP(
            self.in_nf + self.h1 + self.h2, 64, 64, 64, self.out_nf
        )
        # =====================================================================

    def net_propagation_layer_message_fn(self, edges, groundtruth=False):
        if groundtruth:
            last_nf = edges.src["n_atslew"]
        else:
            last_nf = edges.src["new_nf"]

        x = torch.cat([last_nf, edges.src["nf"], edges.dst["nf"]], dim=1)
        x = self.Net_Propogation_Layer_MLP(x)
        return {"efn": x}

    def cell_propagation_layer_message_fn(self, edges, groundtruth=False):
        """this func applied for `cell_out` edges (https://i.imgur.com/EGcAnJy.png)"""

        # generate lut axis query
        if groundtruth:
            last_nf = edges.src["n_atslew"]
        else:
            last_nf = edges.src["new_nf"]

        q = torch.cat([last_nf, edges.src["nf"], edges.dst["nf"]], dim=1)
        q = self.Cell_Propagation_Layer_NLDM_Query_MLP(q)
        q = q.reshape(-1, 2)  # (c->Z Query LUT.x, c->Z Query LUT.y)

        # answer lut axis query
        axis_len = self.in_cell_num_luts * (1 + 2 * self.in_cell_lut_sz)
        axis = edges.data["ef"][:, :axis_len]
        axis = axis.reshape(-1, 1 + 2 * self.in_cell_lut_sz)
        axis = axis.repeat(1, self.lut_dup).reshape(-1, 1 + 2 * self.in_cell_lut_sz)
        a = self.Cell_Propagation_Layer_LUT_Mask_MLP(torch.cat([q, axis], dim=1))

        # transform answer to answer mask matrix
        a = a.reshape(-1, 2, self.in_cell_lut_sz)
        ax, ay = torch.split(a, [1, 1], dim=1)

        # [Debug] a.shape = [#num, 2, 7]
        # [Debug] ax.shape = [#num, 1, 7]
        # [Debug] ay.shape = [#num, 1, 7]

        # [Debug] ax.reshape(-1, self.in_cell_lut_sz, 1).shape = [#num, 7, 1]
        # [Debug] ay.reshape(-1, 1, self.in_cell_lut_sz).shape = [#num, 1, 7]
        a = torch.matmul(
            ax.reshape(-1, self.in_cell_lut_sz, 1),
            ay.reshape(-1, 1, self.in_cell_lut_sz),
        )  # batch tensor product
        # [Debug] a.shape = [#num, 7, 7]

        # look up answer matrix in lut
        tables_len = self.in_cell_num_luts * self.in_cell_lut_sz**2
        tables = edges.data["ef"][:, axis_len : axis_len + tables_len]

        # [Debug] tables.reshape(-1, 1, 1, self.in_cell_lut_sz**2).shape = [#num, 1, 1, 49]
        # [Debug] a.reshape(-1, 4, self.in_cell_lut_sz**2, 1).shape = [#num, 4, 49, 1]
        # https://i.imgur.com/qSw6qH6.png
        r = torch.matmul(
            tables.reshape(-1, 1, 1, self.in_cell_lut_sz**2),
            a.reshape(-1, 4, self.in_cell_lut_sz**2, 1),
        )  # batch dot product

        # construct final msg
        r = r.reshape(len(edges), self.in_cell_num_luts * self.lut_dup)
        x = torch.cat([last_nf, edges.src["nf"], edges.dst["nf"], r], dim=1)
        x = self.Cell_Propagation_Layer_Cellprop_MLP(x)
        k, f1, f2, cef = torch.split(x, [1, self.h1, self.h2, self.out_cef], dim=1)
        k = torch.sigmoid(k)
        return {"efc1": f1 * k, "efc2": f2 * k, "efce": cef}

    def cell_propagation_layer_reduce_fn(self, nodes):
        """This function gathers all edge messages, pass through a MLP
        the final node feature `new_nf` is used as predicted arrival time and slew

        :param nodes: The nodes should be `output_nodes_nonpi` (output nodes but not primary input nodes)
        :return: _description_
        """
        x = torch.cat([nodes.data["nf"], nodes.data["nfc1"], nodes.data["nfc2"]], dim=1)
        x = self.Cell_Propagation_Layer_Cellprop_Reduce_MLP(x)
        return {"new_nf": x}

    def node_skip_level_o(self, nodes):
        return {"new_nf": nodes.data["n_atslew"]}

    def forward(self, g, ts, nf, groundtruth=False):
        assert (
            len(ts["topo"]) % 2 == 0
        ), "The number of logic levels must be even (net, cell, net)"

        with g.local_scope():
            # init level 0 with ground truth features
            g.ndata["nf"] = nf
            g.ndata["new_nf"] = torch.zeros(
                g.num_nodes(), self.out_nf, device="cuda", dtype=nf.dtype
            )
            g.apply_nodes(self.node_skip_level_o, ts["pi_nodes"])

            def net_propagation_layer(nodes, groundtruth):
                """Net Progation layer
                as (X-> c) is the "net_out", thus `self.edge_msg_net` is the implementation of this block https://i.imgur.com/GAoyPgf.png
                This function is apply for `input_nodes` or nodes in 1, 3, 5, 7...th of the topology graph ONLY!

                :param nodes: input nodes (https://i.imgur.com/7n9TlYj.png).
                :param groundtruth: _description_
                """

                g.pull(
                    nodes,
                    functools.partial(
                        self.net_propagation_layer_message_fn, groundtruth=groundtruth
                    ),
                    fn.sum("efn", "new_nf"),
                    etype="net_out",
                )

            def cell_propagation_layer(nodes, groundtruth):
                """_summary_

                :param nodes: The nodes should be `output_nodes_nonpi` (output nodes but not primary input nodes)
                :param groundtruth: _description_
                """

                # (doc) Return the incoming edges of the given nodes.
                # (me) as nodes are output nodes -> es is the `cell_out` edges
                es = g.in_edges(nodes, etype="cell_out")

                g.apply_edges(
                    functools.partial(
                        self.cell_propagation_layer_message_fn, groundtruth=groundtruth
                    ),
                    es,
                    etype="cell_out",
                )

                g.send_and_recv(
                    es,
                    fn.copy_e("efc1", "efc1"),
                    fn.sum("efc1", "nfc1"),
                    etype="cell_out",
                )
                g.send_and_recv(
                    es,
                    fn.copy_e("efc2", "efc2"),
                    fn.max("efc2", "nfc2"),
                    etype="cell_out",
                )
                g.apply_nodes(self.cell_propagation_layer_reduce_fn, nodes)

            if groundtruth:
                # don't need to propagate.
                # propogate the groundtruth `n_atslew`
                net_propagation_layer(ts["input_nodes"], groundtruth=True)

                # propogate the groundtruth `n_atslew`
                cell_propagation_layer(ts["output_nodes_nonpi"], groundtruth=True)

            else:
                # propagate
                for i in range(1, len(ts["topo"])):
                    if i % 2 == 1:
                        net_propagation_layer(ts["topo"][i], groundtruth)
                    else:
                        cell_propagation_layer(ts["topo"][i], groundtruth)

            return (
                g.edges["cell_out"].data["efce"],  # (predicted cell delays)
                g.ndata[
                    "new_nf"
                ],  # https://i.imgur.com/GAoyPgf.png (predicted arrival time and slew)
            )


class TimingGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nc1 = NetConv(10, 2, 32)
        self.nc2 = NetConv(32, 2, 32)
        self.nc3 = NetConv(
            32, 2, 16
        )  # 16 = 4x delay + 12x arbitrary (might include cap, beta)
        self.prop = SignalProp(
            in_nf=10 + 16, in_cell_num_luts=8, in_cell_lut_sz=7, out_nf=8, out_cef=4
        )

    def forward(self, g, ts, groundtruth=False):
        nf0 = g.ndata["nf"]

        # net delay prediction
        x = self.nc1(g, ts, nf0)
        x = self.nc2(g, ts, x)
        x = self.nc3(g, ts, x)
        net_delays = x[:, :4]
        # nf0.shape = [#nodes, 10]
        # x.shape = [#nodes, 16]

        # cell delay prediction
        # ---------------------------------
        # combination of original feature + predicted net delay
        nf1 = torch.cat([nf0, x], dim=1)
        cell_delays, atslew = self.prop(g, ts, nf1, groundtruth=groundtruth)

        # atslew.shape = [#nodes, 8]
        # cell_delays = [#cells, 4]
        return net_delays, cell_delays, atslew
