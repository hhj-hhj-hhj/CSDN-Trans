import copy
import torch
import torchvision
import torch.nn as nn
from .gem_pool import GeneralizedMeanPoolingP


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Classifier(nn.Module):
    def __init__(self, pid_num):
        super(Classifier, self, ).__init__()
        self.pid_num = pid_num
        self.GEM = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(2048)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features_map):
        features = self.GEM(features_map)
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return features, cls_score, self.l2_norm(bn_features)


class Classifier2(nn.Module):
    def __init__(self, pid_num):
        super(Classifier2, self, ).__init__()
        self.pid_num = pid_num
        self.BN = nn.BatchNorm1d(1024)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(1024, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features):
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return cls_score, self.l2_norm(features)


class PromptLearner(nn.Module):
    def __init__(self, num_class, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X X X X person."
        ctx_dim = 512
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class AttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionFusion, self).__init__()
        self.dropout_rate = 0.1
        self.embed_dim = embed_dim
        self.embed_dim_qkv = embed_dim

        self.embedding_q = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim_qkv, kernel_size=1),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate)
        )
        self.embedding_k = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim_qkv, kernel_size=1),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate)
        )
        self.embedding_v = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim_qkv, kernel_size=1),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate)
        )
        self.embedding_common = nn.Conv2d(self.embed_dim_qkv, self.embed_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 学习的标量，用于控制注意力机制的输出

    def q_k_v_product_attention(self, q_emb, k_emb, v_emb):
        # Flatten the spatial dimensions for batch matrix multiplication
        batch_size, channels, height, width = q_emb.size()
        q_flat = q_emb.view(batch_size, channels, -1)
        k_flat = k_emb.view(batch_size, channels, -1).permute(0, 2, 1)
        v_flat = v_emb.view(batch_size, channels, -1)

        # Compute attention weights
        weights = torch.bmm(q_flat, k_flat)
        weights = torch.div(weights, (self.embed_dim_qkv ** 0.5))
        weights = self.softmax(weights)

        # Weighted sum of values
        new_v_flat = torch.bmm(weights, v_flat)
        new_v_emb = new_v_flat.view(batch_size, channels, height, width)
        return new_v_emb

    def forward(self, shape_map, img_map):
        q_emb = self.embedding_q(shape_map)
        k_emb = self.embedding_k(img_map)
        v_emb = self.embedding_v(img_map)
        new_v_emb = self.q_k_v_product_attention(q_emb, k_emb, v_emb)
        new_feature_map = self.embedding_common(new_v_emb)
        new_feature_map = self.gamma * new_feature_map + img_map
        return new_feature_map


class SelfAttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionFusion, self).__init__()
        self.in_channels = in_channels  # 设定通道数
        self.dropout_rate = 0.1  # 设定dropout率
        self.query_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate)
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate)
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate)
        )
        self.gamma = nn.Parameter(torch.zeros(1))  # 学习融合权重
        # self.gamma = nn.Parameter(torch.tensor(0.5))  # 学习融合权重

    def forward(self, feature_shape, feature_orig):
        batch_size, C, height, width = feature_orig.size()

        # 计算 Q, K, V
        Q = self.query_conv(feature_orig).view(batch_size, C, -1)  # (batch_size, C, H*W)
        K = self.key_conv(feature_shape).view(batch_size, C, -1)  # (batch_size, C, H*W)
        V = self.value_conv(feature_shape).view(batch_size, C, -1)  # (batch_size, C, H*W)

        # 计算注意力权重
        scaled_attention_logits = torch.bmm(Q, K.permute(0, 2, 1)) / (self.in_channels ** 0.5)
        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)  # (batch_size, H*W, H*W)

        # 计算加权特征
        attention_out = torch.bmm(attention_weights, V).view(batch_size, C, height, width)  # (batch_size, C, H, W)

        # 融合特征
        fused_feature = self.gamma * attention_out + feature_orig
        return fused_feature


class Model(nn.Module):
    def __init__(self, num_classes, img_h, img_w):
        super(Model, self).__init__()
        self.in_planes = 2048
        self.num_classes = num_classes

        self.h_resolution = int((img_h - 16) // 16 + 1)
        self.w_resolution = int((img_w - 16) // 16 + 1)
        self.vision_stride_size = 16
        clip_model = load_clip_to_cpu('RN50', self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder1 = nn.Sequential(clip_model.visual.conv1, clip_model.visual.bn1, clip_model.visual.conv2,
                                            clip_model.visual.bn2, clip_model.visual.conv3, clip_model.visual.bn3,
                                            clip_model.visual.relu, clip_model.visual.avgpool)
        self.image_encoder2 = copy.deepcopy(self.image_encoder1)

        self.image_encoder = nn.Sequential(clip_model.visual.layer1, clip_model.visual.layer2, clip_model.visual.layer3,
                                           clip_model.visual.layer4)
        self.attnpool = clip_model.visual.attnpool
        self.classifier = Classifier(self.num_classes)
        self.classifier2 = Classifier2(self.num_classes)

        self.prompt_learner = PromptLearner(num_classes, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)
        self.image_attention_fusion = SelfAttentionFusion(self.in_planes)
        self.text_features_p, self.text_features_n = self.get_normal_text_features(clip_model)

    def get_normal_text_features(self, clip_model):
        text_p = "A photo of a person"
        text_n = "Backgrounds for people's photo"
        text_tokens_p = clip.tokenize([text_p]).cuda()
        text_tokens_n = clip.tokenize([text_n]).cuda()
        with torch.no_grad():
            text_features_p = clip_model.encode_text(text_tokens_p)
            text_features_n = clip_model.encode_text(text_tokens_n)
        return text_features_p, text_features_n

    def forward(self, x1=None, x2=None, shape_img=None, label=None, get_image=False, get_text=False, img_map=None,shape_map=None, \
                fusion_map=None, get_map=False, get_atten=False, maps2feature=False):
        if get_image == True:
            if x1 is not None and x2 is None:
                image_features_map1 = self.image_encoder1(x1)
                image_features_map1 = self.image_encoder(image_features_map1)
                image_features1_proj = self.attnpool(image_features_map1)[0]
                return image_features1_proj
            elif x1 is None and x2 is not None:
                image_features_map2 = self.image_encoder2(x2)
                image_features_map2 = self.image_encoder(image_features_map2)
                image_features2_proj = self.attnpool(image_features_map2)[0]
                return image_features2_proj

        if get_text == True:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_map == True:
            if x1 is not None and x2 is None:
                image_features_map1 = self.image_encoder1(x1)
                image_features_map1 = self.image_encoder(image_features_map1)
                return image_features_map1
            elif x1 is None and x2 is not None:
                image_features_map2 = self.image_encoder2(x2)
                image_features_map2 = self.image_encoder(image_features_map2)
                return image_features_map2
        if get_atten == True:
            image_features_maps = self.image_attention_fusion(shape_map, img_map)
            return image_features_maps

        if maps2feature == True:
            image_features_proj = self.attnpool(fusion_map)[0]
            return image_features_proj

        if x1 is not None and x2 is not None:

            image_features_map1 = self.image_encoder1(x1)
            image_features_map2 = self.image_encoder2(x2)
            image_features_maps = torch.cat([image_features_map1, image_features_map2], dim=0)
            image_features_maps = self.image_encoder(image_features_maps)
            image_features_proj = self.attnpool(image_features_maps)[0]
            features, cls_scores, _ = self.classifier(image_features_maps)
            cls_scores_proj, _ = self.classifier2(image_features_proj)

            pp = None
            # B, C, H, W = image_features_maps.shape
            # pp = image_features_maps.view(B, 8, self.in_planes // 8, 6, H // 6, W)
            # pp = pp.mean(-1).mean(-1).permute(0, 1, 3, 2).contiguous()
            # pp = pp.view(B, 8 * 6, self.in_planes // 8)
            return [features, image_features_proj], [cls_scores, cls_scores_proj], pp

        elif x1 is not None and x2 is None:

            image_features_map1 = self.image_encoder1(x1)
            image_features_map1 = self.image_encoder(image_features_map1)
            image_features1_proj = self.attnpool(image_features_map1)[0]
            _, _, test_features1 = self.classifier(image_features_map1)
            _, test_features1_proj = self.classifier2(image_features1_proj)

            return torch.cat([test_features1, test_features1_proj], dim=1)

        elif x1 is None and x2 is not None:

            image_features_map2 = self.image_encoder2(x2)
            image_features_map2 = self.image_encoder(image_features_map2)
            image_features2_proj = self.attnpool(image_features_map2)[0]
            _, _, test_features2 = self.classifier(image_features_map2)
            _, test_features2_proj = self.classifier2(image_features2_proj)

            return torch.cat([test_features2, test_features2_proj], dim=1)


from .clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model
