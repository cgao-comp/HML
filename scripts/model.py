# An example for node and feature contrastive learning via pruning.

class NodeMMCLModel(nn.Module):
    def __init__(
        self,
        feature,
        temperature=args.temperature,
        mm_embedding=args.mm_embedding_dim,
        node_embedding=args.node_embedding_dim,
    ):
        super().__init__()

        # The pruned mm encoder: some parameters will first be masked at the main function.
        self.mm_encoder = MMEncoder() 
        self.mm_projection = ProjectionHead(embedding_dim=mm_embedding)

        # Projecting all features in heterogeneous graph to the same space
        self.feature_projection = FeatureProjection(feature)

        # The non-pruned node encoder
        self.node_encoder = NodeEncoder()
        self.node_embed_projection = ProjectionHead(embedding_dim=node_embedding)

        self.feature_projection_prune = FeatureProjection(feature)

        # The pruned node encoder
        self.node_encoder_prune = NodeEncoder()
        self.node_embed_prune_projection = ProjectionHead(embedding_dim=node_embedding)

        self.temperature = temperature

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.temperature)
        return sim_matrix

    def forward(self, batch,feature,adj,pos):

        # Getting mm features
        mm_features = self.mm_encoder(feature['feature'][batch['entity']])
        batch = {k: v.to(args.device) for k, v in batch.items() if k != "mm"}
        mm_embeddings = self.mm_projection(mm_features)
        
        # ocr_features = 
        # ocr_embeddings = self.ocr_projection(ocr_features)
        # batch = {k: v.to(args.device) for k, v in batch.items() if k != "mm"}

        # Getting pruned mm embeddings
        # mm_embeddings = self.mm_projection(mm_features)

        # Getting pruned node embeddings
        features = self.feature_projection_prune(feature)
        node_embed_prune = self.node_encoder_prune(features.float(), adj)
        node_embeddings_project_prune = self.node_embed_prune_projection(node_embed_prune)
        # print(batch)
        node_embedding_batch_prune = node_embeddings_project_prune[batch['entity']]


        logits_prune = (node_embedding_batch_prune @ mm_embeddings.T) / self.temperature
        mm_similarity = mm_embeddings @ mm_embeddings.T
        nodes_similarity_prune = node_embedding_batch_prune @ node_embedding_batch_prune.T
        # print(mm_similarity.size())
        # print(nodes_similarity_prune.size())
        targets = F.softmax(
            (mm_similarity + nodes_similarity_prune) / 2 * self.temperature, dim=-1
        )
        nodes_loss = cross_entropy(logits_prune, targets, reduction='none')
        mm_loss = cross_entropy(logits_prune.T, targets.T, reduction='none')
        loss_inter = (mm_loss + nodes_loss) / 2.0  # shape: (batch_size)

        # Getting non-pruned node embeddings
        features = self.feature_projection(feature)
        node_embed = self.node_encoder(features.float(), adj)
        node_embeddings_project = self.node_embed_projection(node_embed)
        node_embedding_batch = node_embeddings_project[batch['entity']]

        if args.pos:
            matrix_prune2non = self.sim(node_embedding_batch,node_embedding_batch_prune)
            matrix_prune2non = matrix_prune2non / (torch.sum(matrix_prune2non, dim=1).view(-1, 1) + 1e-8)
            pos_d = pos.to_dense()
            pos_index = batch['entity'].sort()
            pos_d_batch = pos_d[pos_index[0]][:,pos_index[0]]
            lori_prune = -torch.log(matrix_prune2non.mul(pos_d_batch).sum(dim=-1)).mean()

            matrix_non2prune = matrix_prune2non.t()
            matrix_non2prune = matrix_non2prune / (torch.sum(matrix_non2prune, dim=1).view(-1, 1) + 1e-8)
            lori_non = -torch.log(matrix_non2prune.mul(pos_d_batch).sum(dim=-1)).mean()

            loss_intra = (lori_non + lori_prune) / 2.0
            # lbda = 0.5
            loss = loss_inter.mean() + loss_intra

        else:
            logits_intra = (node_embedding_batch @ node_embedding_batch_prune.T) / self.temperature
            node_similarity_prune = node_embedding_batch_prune @ node_embedding_batch_prune.T

            nodes_similarity = node_embedding_batch @ node_embedding_batch.T

            targets_intra = F.softmax(
                (node_similarity_prune + nodes_similarity) / 2 * self.temperature, dim=-1
            )
            nodes_loss_intra = cross_entropy(logits_intra, targets_intra, reduction='none')
            node_loss_prune_intra = cross_entropy(logits_intra.T, targets_intra.T, reduction='none')
            loss_intra = (nodes_loss_intra + node_loss_prune_intra) / 2.0  # shape: (batch_size)
            loss = loss_inter.mean() + loss_intra.mean()
        # loss = loss_intra.mean()
        return loss,node_embed_prune,node_embed
