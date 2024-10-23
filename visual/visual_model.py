import copy
import torch
import torchvision
import torch.nn as nn
from network.gem_pool import GeneralizedMeanPoolingP


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

        self.embedding_q = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim_qkv),
                                         nn.Tanh(), nn.Dropout(self.dropout_rate))
        self.embedding_k = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim_qkv),
                                         nn.Tanh(), nn.Dropout(self.dropout_rate))
        self.embedding_v = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim_qkv),
                                         nn.Tanh(), nn.Dropout(self.dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(self.embed_dim_qkv, self.embed_dim))
        self.softmax = nn.Softmax(dim=1)

    def q_k_v_product_attention(self, q_emb, k_emb, v_emb):
        weights = torch.bmm(q_emb, k_emb.permute(0, 2, 1))
        weights = torch.div(weights, (self.embed_dim_qkv ** 0.5))
        weights = self.softmax(weights)
        new_v_emb = weights.bmm(v_emb)
        return new_v_emb

    def forward(self, text_features1, text_features2):
        batch_size = text_features1.size(0)
        q_emb = self.embedding_q(text_features1.unsqueeze(1))
        k_emb = self.embedding_k(text_features2.unsqueeze(1))
        v_emb = self.embedding_v(text_features2.unsqueeze(1))
        new_v_emb = self.q_k_v_product_attention(q_emb, k_emb, v_emb)
        new_text_features = self.embedding_common(new_v_emb)
        new_text_features = new_text_features.view(batch_size, self.embed_dim) + text_features1
        return new_text_features

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
        K = self.key_conv(feature_shape).view(batch_size, C, -1)    # (batch_size, C, H*W)
        V = self.value_conv(feature_shape).view(batch_size, C, -1)  # (batch_size, C, H*W)

        # 计算注意力权重
        scaled_attention_logits = torch.bmm(Q, K.permute(0,2,1)) / (self.in_channels ** 0.5)
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
        # self.image_attention_fusion = AttentionFusion(self.in_planes)
        self.image_attention_fusion = SelfAttentionFusion(self.in_planes)
        self.text_features_p, self.text_features_n = self.get_normal_text_features(clip_model)

    def get_normal_text_features(self, clip_model):
        text_p = "A photo of a person"
        text_n = "The background of the photo"
        text_tokens_p = clip.tokenize([text_p]).cuda()
        text_tokens_n = clip.tokenize([text_n]).cuda()
        with torch.no_grad():
            text_features_p = clip_model.encode_text(text_tokens_p)
            text_features_n = clip_model.encode_text(text_tokens_n)
        return text_features_p, text_features_n


    def forward(self, data=None):
        import cv2
        import numpy as np
        from PIL import Image
        shape_path = r"E:\hhj\SYSU-MM01-output\cam1\0001\rgb_0001.png"
        shape = cv2.imread(shape_path)
        shape_np = np.array(shape)
        shape_np[np.any(shape_np != [0, 0, 0], axis=-1)] = [255, 255, 255]
        shape = Image.fromarray(shape_np)
        from torchvision import transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_test_rgb = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((288, 144)),
            transforms.ToTensor(),
            normalize])
        shape_tensor = transform_test_rgb(shape.copy())
        shape_tensor = shape_tensor.unsqueeze(0)
        shape_tensor = shape_tensor.to('cuda')

        img = data
        shape = shape_tensor
        img_map = self.image_encoder1(img)
        img_map = self.image_encoder(img_map)
        shape_map = self.image_encoder1(shape)
        shape_map = self.image_encoder(shape_map)
        image_features_maps = self.image_attention_fusion( img_map,shape_map)
        # image_features_maps = img_map
        image_features_proj = self.attnpool(image_features_maps)[0]
        return image_features_proj


from network.clip import clip


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

def is_same(a, b):
    for i in range(len(a)):
        if not torch.all(a[i] == b[i]):
            print(f'index {i} is not same')

class PromptLearner_share(nn.Module):
    def __init__(self, num_class, dtype, token_embedding):
        super().__init__()
        rgb_ctx_init = "A visible photo of a X X X X person."
        ir_ctx_init = "A infrared photo of a X X X X person."
        ctx_init = "A photo of a X X X X person."
        ctx_dim = 512

        rgb_n_ctx = 5
        ir_n_ctx = 5
        n_ctx = 4

        rgb_tokenized_prompts = clip.tokenize(rgb_ctx_init).cuda()
        ir_tokenized_prompts = clip.tokenize(ir_ctx_init).cuda()
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            rgb_embedding = token_embedding(rgb_tokenized_prompts).type(dtype)
            ir_embedding = token_embedding(ir_tokenized_prompts).type(dtype)
            embedding = token_embedding(tokenized_prompts).type(dtype)
        print(f'embeeding shape: {embedding.shape}')
        self.rgb_tokenized_prompts = rgb_tokenized_prompts
        self.ir_tokenized_prompts = ir_tokenized_prompts
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        self.register_buffer("rgb_token_prefix", rgb_embedding[:, :rgb_n_ctx + 1, :])
        self.register_buffer("ir_token_prefix", ir_embedding[:, :ir_n_ctx + 1, :])
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("rgb_token_suffix", rgb_embedding[:, rgb_n_ctx + 1 + n_cls_ctx:, :])
        self.register_buffer("ir_token_suffix", ir_embedding[:, ir_n_ctx + 1 + n_cls_ctx:, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, mode = None):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        if mode == 'rgb':
            prefix = self.rgb_token_prefix.expand(b, -1, -1)
            suffix = self.rgb_token_suffix.expand(b, -1, -1)
        elif mode == 'ir':
            prefix = self.ir_token_prefix.expand(b, -1, -1)
            suffix = self.ir_token_suffix.expand(b, -1, -1)
        else:
            prefix = self.token_prefix.expand(b, -1, -1)
            suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (b, rgb_n_ctx/ir_n_ctx/n_ctx, dim)
                cls_ctx,  # (b, n_cls_ctx, dim)
                suffix,  # (b, *, dim)
            ],
            dim=1,
        )
        return prompts


class PromptLearner_share_with_cloth(nn.Module):
    def __init__(self, num_class, dtype, token_embedding):
        super().__init__()
        rgb_ctx_init = "A visible photo of a X X X X person wearing Y Y Y Y."
        ir_ctx_init = "A infrared photo of a X X X X person."
        ctx_init = "A photo of a X X X X person wearing Y Y Y Y."
        ctx_dim = 512

        rgb_n_ctx = 5
        ir_n_ctx = 5
        mid_n_ctx = 2
        n_ctx = 4

        rgb_tokenized_prompts = clip.tokenize(rgb_ctx_init).cuda()
        ir_tokenized_prompts = clip.tokenize(ir_ctx_init).cuda()
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            rgb_embedding = token_embedding(rgb_tokenized_prompts).type(dtype)
            ir_embedding = token_embedding(ir_tokenized_prompts).type(dtype)
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.rgb_tokenized_prompts = rgb_tokenized_prompts
        self.ir_tokenized_prompts = ir_tokenized_prompts
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cloth_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        cloth_cls_vectors = torch.empty(num_class, cloth_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        nn.init.normal_(cloth_cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)
        self.cloth_cls_ctx = nn.Parameter(cloth_cls_vectors)

        # print('embedding 对比')
        # print(f'embedding shape: {embedding.shape}')
        # is_same(rgb_embedding[0], ir_embedding[0])

        self.register_buffer("rgb_token_prefix", rgb_embedding[:, :rgb_n_ctx + 1, :])
        self.register_buffer("ir_token_prefix", ir_embedding[:, :ir_n_ctx + 1, :])
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])

        self.register_buffer("rgb_token_mid", rgb_embedding[:, rgb_n_ctx + n_cls_ctx + 1: rgb_n_ctx + n_cls_ctx + mid_n_ctx + 1, :])
        self.register_buffer("token_mid", embedding[:, n_ctx + n_cls_ctx + 1: n_ctx + n_cls_ctx + mid_n_ctx + 1, :])

        self.register_buffer("rgb_token_suffix", rgb_embedding[:, rgb_n_ctx + n_cls_ctx + mid_n_ctx + 1 + cloth_cls_ctx:, :])
        self.register_buffer("ir_token_suffix", ir_embedding[:, ir_n_ctx + 1 + n_cls_ctx:, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + n_cls_ctx + mid_n_ctx + 1 + cloth_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, mode=None):
        cls_ctx = self.cls_ctx[label]
        cloth_cls_ctx = self.cloth_cls_ctx[label]
        b = label.shape[0]
        if mode == 'ir':
            prefix = self.ir_token_prefix.expand(b, -1, -1)
            suffix = self.ir_token_suffix.expand(b, -1, -1)
            prompts = torch.cat(
                [
                    prefix,  # (b, rgb_n_ctx/ir_n_ctx/n_ctx, dim)
                    cls_ctx,  # (b, n_cls_ctx, dim)
                    suffix,  # (b, *, dim)
                ],
                dim=1,
            )
            return prompts
        elif mode == 'rgb':
            prefix = self.rgb_token_prefix.expand(b, -1, -1)
            mid = self.rgb_token_mid.expand(b, -1, -1)
            suffix = self.rgb_token_suffix.expand(b, -1, -1)
        else:
            prefix = self.token_prefix.expand(b, -1, -1)
            mid = self.token_mid.expand(b, -1, -1)
            suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (b, rgb_n_ctx/ir_n_ctx/n_ctx, dim)
                cls_ctx,  # (b, n_cls_ctx, dim)
                mid,  # (b, mid_n_ctx, dim)
                cloth_cls_ctx,  # (b, cloth_cls_ctx, dim)
                suffix,  # (b, *, dim)
            ],
            dim=1,
        )
        return prompts


clip_model = load_clip_to_cpu('RN50', 18, 9, 16)
clip_model.to("cuda")
prompter = PromptLearner_share(395, clip_model.dtype, clip_model.token_embedding).cuda()
text_encoder = TextEncoder(clip_model).cuda()
label = torch.tensor([0], device='cuda')
prompt_rgb = prompter(label, mode='rgb')
prompt_ir = prompter(label, mode='ir')
text_features1 = text_encoder(prompt_rgb, prompter.rgb_tokenized_prompts)
text_features2 = text_encoder(prompt_ir, prompter.ir_tokenized_prompts)
att = AttentionFusion(1024).cuda()
fusion = att(text_features1, text_features2)
print('hello')
# rgb = prompt(label=label, mode='rgb')[0]
# ir = prompt(label=label, mode='ir')[0]
# normal = prompt(label=label, mode='common')[0]
# print('---------------')
# nor_rgb = torch.cat((rgb[:6], rgb[10:12], rgb[16:]))
# nor_normal = torch.cat((normal[:3], normal[2:5], normal[9:11], normal[15:]))
#
# is_same(nor_rgb, nor_normal)


