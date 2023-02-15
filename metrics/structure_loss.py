from .base import Metrics
import torch


class StructureLoss(Metrics):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def compute(self, output, pts3d, batch):
        def get_union(left_tensor, right_tensor):
            left_tensor_tmp = left_tensor.unsqueeze(1)
            right_tensor_tmp = right_tensor.unsqueeze(0)
            mask_all = left_tensor_tmp == right_tensor_tmp
            samemask = mask_all.all(-1).nonzero()  # same and repeated edges
            selfmask = torch.logical_not(
                mask_all.all(-1).any(1)
            )  # edges only in left tensor
            diffmask = (
                torch.logical_not(mask_all).any(-1).all(0)
            )  # edges only in right tesnor
            same_edges = left_tensor[samemask[:, 0], :]
            diff_edges = right_tensor[diffmask]
            union = torch.vstack([same_edges, left_tensor[selfmask], diff_edges])
            return union, samemask, selfmask, diffmask

        def get_distribution_loss(nodes, edge_union, edge_union2, pred_union, gt_union):
            u1_sort_index_column0 = edge_union[:, 0].argsort()
            u2_sort_index_column0 = edge_union2[:, 0].argsort()
            edge_union, edge_union2 = (
                edge_union[u1_sort_index_column0],
                edge_union2[u2_sort_index_column0],
            )
            pred_union = pred_union[u1_sort_index_column0]
            gt_union = gt_union[u2_sort_index_column0]
            loss = 0

            for n in range(pred_union.shape[0]):
                center1 = pred_union[n]
                ind1 = edge_union[:, 0] == edge_union[n][0]
                neighbor = pred_union[ind1]
                neighbor_edges = edge_union[ind1]
                sort_ind1 = neighbor_edges[:, 1].argsort()
                neighbor1 = neighbor[sort_ind1]
                d1 = center1 - neighbor1
                # sim1 = (d1**2).sum(-1).sqrt()
                # s1 = sim1.exp() / (sim1.exp().sum() + 1e-4)
                s1 = torch.max(d1, dim=-1)[0]

                center2 = gt_union[n]
                ind2 = edge_union2[:, 0] == edge_union[n][0]
                neighbor2 = gt_union[ind2]
                neighbor_edges2 = edge_union2[ind2]
                sort_ind2 = neighbor_edges2[:, 1].argsort()
                neighbor2 = neighbor2[sort_ind2]
                d2 = center2 - neighbor2
                # sim2 = (d2**2).sum(-1).sqrt()
                # s2 = sim2.exp() / (sim2.exp().sum() + 1e-4)
                s2 = torch.max(d2, dim=-1)[0]

                # loss += (s1 * torch.log(s1 / s2)).mean()
                loss += ((s1 - s2) ** 2).mean()
            return loss / pred_union.shape[0]

        idx_pairs1 = batch["idx_pairs"][0]
        idx_pairs2 = batch["idx_pairs"][1]
        gt = batch["rel_features"][1]
        batch_size, num_rel = output.shape[:2]
        loss = 0
        for b in range(batch_size):
            # remove unseen nodes
            idx_pairs1_tmp = idx_pairs1[b]
            id2 = idx_pairs2[b]
            mask = torch.isin(id2, idx_pairs1_tmp).all(-1)
            idx_pairs2_tmp = id2[mask]
            gt_tmp = gt[b][mask]
            nodes = torch.unique(idx_pairs1_tmp)

            """ 
            make virtual connections
            """
            # u, _ = torch.unique(idx_pairs1_tmp, dim=0, return_inverse=True)
            # print(u.shape)
            edge_union, samemask, selfmask, diffmask = get_union(
                idx_pairs1_tmp, idx_pairs2_tmp
            )
            pred_union = torch.vstack(
                [output[b][samemask[:, 0]], output[b][selfmask], gt_tmp[diffmask]]
            )

            edge_union2, samemask2, selfmask2, diffmask2 = get_union(
                idx_pairs2_tmp, idx_pairs1_tmp
            )
            gt_union = torch.vstack(
                [gt_tmp[samemask2[:, 0]], gt_tmp[selfmask2], output[b][diffmask2]]
            )
            if not (
                (edge_union.unsqueeze(0) == edge_union2.unsqueeze(1))
                .any(-1)
                .any(0)
                .all()
            ):
                raise "Union graph not same, algo error!!!"

            """
            All relatinships are predicted in one mode
            """
            if edge_union.shape[0] > 1000:
                continue
            # print(edge_union.shape, edge_union2.shape)
            loss += get_distribution_loss(
                nodes, edge_union, edge_union2, pred_union, gt_union
            )
            """
            align dist1 and dist2
            """
        loss /= batch_size
        return loss

    def __call__(self, output, pts3d, batch):
        return self.compute(output, pts3d, batch)
