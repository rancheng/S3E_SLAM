# Copyright 2021 Ran Cheng <ran.cheng2@mail.mcgill.ca>
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import itertools

import torch
import h5py
import numpy as np
from tqdm import tqdm
from module.ViT import FViT
import os
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import dgl
from dgl.data import DGLDataset
from itertools import product


# dataset for the embedding training
class GNNDataset(Dataset):
    def __init__(
            self,
            data_root, model_cfgs, patch_size="16", iou_edge_th=0.01):
        """
        dataset initialization function, generate the graph data
        :param data_root: dataset root, which contains the torch_data folder
        :param model_cfgs: configures for the VIT model
        :param patch_size: torch patch size, there are three options: ["16", "32", "64"]
        :param iou_edge_th: the threshold for iou filter to build the edge between nodes of pose graph
        """
        self._data_root = data_root
        self._patch_size = patch_size
        self._subgraph_size = 5
        self._patch_key = "patch_{}".format(self._patch_size)
        assert self._patch_key in ['patch_16', 'patch_32', 'patch_64']
        self._torch_data_root = os.path.join(data_root, "torch_data")
        self._pose_data_root = os.path.join(data_root, "pose")
        assert os.path.exists(self._torch_data_root), f"{self._torch_data_root} does not exist"
        self._torch_data_files = glob.glob(os.path.join(self._torch_data_root, "*.h5"))
        self._torch_data_files = sorted(self._torch_data_files)
        self._dataset_size = len(self._torch_data_files)
        # iou heatmap
        iou_heatmap_fname = os.path.join(self._data_root, "iou_heatmap.npy")
        if os.path.isfile(iou_heatmap_fname):
            # load the query map of iou data
            self._iou_map = np.load(iou_heatmap_fname)
        else:
            raise Exception("IoU heatmap is empty! Please train the embedding first.")
        # collect edge data for the pose graph
        self._g_edge_list = []
        self._iou_edge_th = iou_edge_th
        # cook the node embeddings offline
        self.emb_data_buffer = None
        self._node_embedding_dir = os.path.join(self._data_root, "embeddings")
        if not os.path.exists(self._node_embedding_dir):
            os.mkdir(self._node_embedding_dir)
        if not len(os.listdir(self._node_embedding_dir)) > 1:
            # embed the frames with vit
            print("no embedding detected!")
            self.embed_graph_nodes(model_cfgs)
        else:
            print("embedding data found, loading the embedding data!")
            self._embedding_data_files = glob.glob(os.path.join(self._node_embedding_dir, "*.npy"))
            self._embedding_data_files = sorted(self._embedding_data_files)
            self.load_embedding_data()
        print("building graph from IoU map...")
        self.build_graph()
        # construct the masked graph dataset
        self.masked_graph_edge_list = []
        self.emb_data_buffer_graph = None
        self.emb_data_buffer_query = None
        self.graph_idx = None
        self.node_idx = None
        self.graph_iou_map = None
        self.query_iou_map = None
        self.graph_idx_map = {}
        print("masking graph...")
        self.mask_graph()
        # self.train_query_idx, self.val_query_idx = train_test_split(np.arange(len(self.node_idx)), shuffle=True)

    def mask_graph(self):
        # mask the graph so that the map graph does not contain the query node
        idx_list = np.arange(len(self.emb_data_buffer))
        # the portion of graph idx shold be large enough, to decrease the occurrence of orphan sub-graphs
        self.graph_idx, self.node_idx = train_test_split(idx_list, train_size=0.9, shuffle=True)
        # filter out the embedding and edges for graph
        self.graph_idx = sorted(self.graph_idx)
        # keep record the index mapping
        for gid, gd in enumerate(self.graph_idx):
            self.graph_idx_map[gd] = gid
        gidx_permute = np.array([list(r) for r in product(self.graph_idx, self.graph_idx)])
        self.graph_iou_map = np.reshape(self._iou_map[gidx_permute[:, 0], gidx_permute[:, 1]],
                                        (len(self.graph_idx), len(self.graph_idx)))
        qidx_permute = np.array([list(r) for r in product(self.node_idx, self.graph_idx)])
        self.query_iou_map = np.reshape(self._iou_map[qidx_permute[:, 0], qidx_permute[:, 1]],
                                        (len(self.node_idx), len(self.graph_idx)))
        print("building masked graph edge ...")
        for edge_item in tqdm(self._g_edge_list):
            if edge_item[0] not in self.node_idx and edge_item[1] not in self.node_idx:
                self.masked_graph_edge_list.append([self.graph_idx_map[edge_item[0]], self.graph_idx_map[edge_item[1]]])
        self.emb_data_buffer_graph = self.emb_data_buffer[self.graph_idx]
        self.emb_data_buffer_query = self.emb_data_buffer[self.node_idx]

    def get_graph(self):
        """
        retrieve the graph data
        :return: masked_edge: array of edge, emb_data_buffer_graph: graph node embedding, edge_features: concatenated node embeddings
        """
        masked_edge = np.array(self.masked_graph_edge_list)
        edge_features = np.linalg.norm(
            self.emb_data_buffer_graph[masked_edge[:, 0], :] - self.emb_data_buffer_graph[masked_edge[:, 1], :], axis=1)
        return torch.from_numpy(masked_edge), torch.from_numpy(self.emb_data_buffer_graph), torch.from_numpy(edge_features)

    def load_embedding_data(self):
        # return the embedding and edges
        self.emb_data_buffer = np.zeros(
            (len(self._embedding_data_files), np.load(self._embedding_data_files[0]).shape[0]))
        for emb_idx, emb_item in enumerate(self._embedding_data_files):
            self.emb_data_buffer[emb_idx] = np.load(emb_item)

    def build_graph(self):
        # build the graph data from the items based on the heatmap
        # | x - - - - x |
        # | - x - - x - |
        # | - - x x - - |
        # | x - - x x - |
        # | - - - - x x |
        # | - x - - - x |
        # search each row and build the edge when the iou exceed the threshold
        row, column = self._iou_map.shape  # row == column since it's fully connected graph
        for ri in range(row):
            ri_connections = list(np.where(self._iou_map[ri] >= self._iou_edge_th)[0])
            if len(ri_connections) < 1:
                self._g_edge_list.append([ri, ri])
            else:
                for ci in ri_connections:
                    self._g_edge_list.append([ri, ci])

    def __len__(self):
        return len(self.node_idx)

    def __getitem__(self, i):
        # return the query node information
        data_collections = {}
        node_idx = self.node_idx[i]
        data_collections['idx'] = node_idx  # query node idx in original sequence
        data_collections['emb_vec'] = self.emb_data_buffer_query[i]
        data_collections['iou_label'] = self.query_iou_map[i]
        return data_collections

    def embed_graph_nodes(self, model_cfgs):
        data_cfg = model_cfgs['data']
        model_cfg = model_cfgs['model']
        ckpt_output_dir = data_cfg['ckpt_output']
        best_model_ckpt_fname = os.path.join(ckpt_output_dir, "weights_epoch_best.pth")
        num_patches = int(model_cfg['num_patches'])
        patch_size = int(model_cfg['patch_size'])
        pos_dim = int(model_cfg['pos_dim'])
        emb_dim = int(model_cfg['emb_dim'])
        code_dim = int(model_cfg['code_dim'])
        depth = int(model_cfg['depth'])
        heads = int(model_cfg['heads'])
        mlp_dim = int(model_cfg['mlp_dim'])
        pool = model_cfg['pool']
        channels = int(model_cfg['channels'])
        dim_head = int(model_cfg['dim_head'])
        dropout = float(model_cfg['dropout'])
        emb_dropout = float(model_cfg['emb_dropout'])
        # device of model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embed_model = FViT(
            num_patches=num_patches,
            patch_size=patch_size,
            pos_dim=pos_dim,
            emb_dim=emb_dim,
            code_dim=code_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,  # rgbd
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        embed_model.to(device)
        pretrained_dict = torch.load(best_model_ckpt_fname)
        model_dict = embed_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        embed_model.load_state_dict(model_dict)
        embed_model.eval()
        # loop and embed
        print("generating the embeddings...")
        with torch.no_grad():
            embedding_list = []
            for td_item in tqdm(self._torch_data_files):
                with h5py.File(td_item) as f:
                    patch_data = torch.tensor(np.expand_dims(np.array(f[self._patch_key][0, :]), axis=0),
                                              dtype=torch.float32).to(device)
                    pos_data = torch.tensor(np.expand_dims(np.array(f['key_points_xyz'][0, :]), axis=0),
                                            dtype=torch.float32).to(device)
                    embedding = embed_model(patch_data, pos_data)
                    embedding = np.squeeze(embedding.cpu().numpy())
                    embedding_fname = td_item.split("/")[-1].replace(".h5", ".npy")
                    embedding_list.append(list(embedding))
                    np.save(os.path.join(self._node_embedding_dir, embedding_fname), embedding)
            self.emb_data_buffer = np.array(embedding_list)


class S3EGNN_Dataset(DGLDataset):
    def __init__(self, edges, node_feats, edge_feats, node_labels):
        super().__init__(name='karate_club')
        self.node_feats = node_feats
        self.node_labels = node_labels
        self.edge_feats = edge_feats
        self.edges = edges

    def process(self):
        node_features = torch.from_numpy(self.node_feats)
        node_labels = torch.from_numpy(self.node_labels)
        edge_features = torch.from_numpy(self.edge_feats)
        edges_src = torch.from_numpy(self.edges[:, 0])
        edges_dst = torch.from_numpy(self.edges[:, 1])

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=self.node_feats.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = self.node_feats.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


if __name__ == "__main__":
    import yaml

    data_root = "/mnt/Data/Datasets/S3E_SLAM/data_generate_1"  # "/mnt/Data/Shared/data_generate_2"
    patch_idx = 12
    m_configs = yaml.safe_load(open("config/vit_config.yaml", 'r'))
    dataset = GNNDataset(data_root, m_configs, str(m_configs['model']['patch_size']))
    data = dataset[30]
    idx = data['idx']
    print(data['emb_vec'].shape)
    print(data['iou_label'].shape)
