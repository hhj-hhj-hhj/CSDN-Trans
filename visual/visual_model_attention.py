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


class PromptLearner_part(nn.Module):
    def __init__(self, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X X X X person's "
        part_list = ['hair', 'face', 'arms', 'legs', 'shoes', 'upper-clothes', 'pants']

        tokenized_prompts_list = [clip.tokenize(ctx_init + part).cuda() for part in part_list]
        with torch.no_grad():
            embedding_list = [token_embedding(tokenized_prompts).type(dtype) for tokenized_prompts in
                              tokenized_prompts_list]

        self.tokenized_prompts_list = tokenized_prompts_list
        self.num_parts = len(part_list)

        for i, embedding in enumerate(embedding_list):
            self.register_buffer(f"token_prefix_{i}", embedding[:, :4, :])
            self.register_buffer(f"token_suffix_{i}", embedding[:, 8:, :])

        print(f'num_parts: {self.num_parts}')

    def forward(self, cls_ctx):
        b = cls_ctx.shape[0]
        prompts = []
        for i in range(self.num_parts):
            prefix = getattr(self, f"token_prefix_{i}").expand(b, -1, -1)
            suffix = getattr(self, f"token_suffix_{i}").expand(b, -1, -1)
            prompt = torch.cat(
                [
                    prefix,  # (b, 4, dim)
                    cls_ctx,  # (b, 4, dim)
                    suffix,  # (b, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.stack(prompts, dim=0)
        # prompts = torch.transpose(prompts, 0, 1)
        return prompts  # (num_parts, b, *, dim)


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
        # print(f'embeeding shape: {embedding.shape}')
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

    def forward(self, label, mode=None):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        if mode == 'get_cls_ctx':
            return cls_ctx
        elif mode == 'rgb':
            prefix = self.rgb_token_prefix.expand(b, -1, -1)
            suffix = self.rgb_token_suffix.expand(b, -1, -1)
        elif mode == 'ir':
            prefix = self.ir_token_prefix.expand(b, -1, -1)
            suffix = self.ir_token_suffix.expand(b, -1, -1)
        elif mode == 'common':
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

        self.register_buffer("rgb_token_prefix", rgb_embedding[:, :rgb_n_ctx + 1, :])
        self.register_buffer("ir_token_prefix", ir_embedding[:, :ir_n_ctx + 1, :])
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])

        self.register_buffer("rgb_token_mid",
                             rgb_embedding[:, rgb_n_ctx + n_cls_ctx + 1: rgb_n_ctx + n_cls_ctx + mid_n_ctx + 1, :])
        self.register_buffer("token_mid", embedding[:, n_ctx + n_cls_ctx + 1: n_ctx + n_cls_ctx + mid_n_ctx + 1, :])

        self.register_buffer("rgb_token_suffix",
                             rgb_embedding[:, rgb_n_ctx + n_cls_ctx + mid_n_ctx + 1 + cloth_cls_ctx:, :])
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

        self.prompt_learner = PromptLearner_share(num_classes, clip_model.dtype, clip_model.token_embedding)
        self.prompt_part = PromptLearner_part(clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x1=None, x2=None, label1=None, label2=None, label=None, get_image=False, get_text=False,
                get_part_text=False):
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
            if label1 is not None and label2 is None:
                prompts1 = self.prompt_learner(label1, mode='rgb')
                text_features1 = self.text_encoder(prompts1, self.prompt_learner.rgb_tokenized_prompts)
                return text_features1
            if label2 is not None and label1 is None:
                prompts2 = self.prompt_learner(label2, mode='ir')
                text_features2 = self.text_encoder(prompts2, self.prompt_learner.ir_tokenized_prompts)
                return text_features2
            elif label is not None:
                prompts_common = self.prompt_learner(label, mode='common')
                text_features_common = self.text_encoder(prompts_common, self.prompt_learner.tokenized_prompts)
                return text_features_common
        if get_part_text == True:
            cls_ctx = self.prompt_learner(self.classifier.cls_ctx[label], mode='get_cls_ctx')
            prompts = self.prompt_part(cls_ctx)
            text_features = []
            for i in range(self.prompt_part.num_parts):
                text_features.append(self.text_encoder(prompts[i], self.prompt_part.tokenized_prompts_list[i]))

            text_features = torch.stack(text_features, dim=0)  # (num_parts, b, dim)
            text_features = text_features.transpose(0, 1)  # (b, num_parts, dim)
            return text_features

        if x1 is not None and x2 is not None:

            image_features_map1 = self.image_encoder1(x1)
            image_features_map2 = self.image_encoder2(x2)
            image_features_maps = torch.cat([image_features_map1, image_features_map2], dim=0)
            image_features_maps = self.image_encoder(image_features_maps)
            image_features_proj = self.attnpool(image_features_maps)[0]
            features, cls_scores, _ = self.classifier(image_features_maps)
            cls_scores_proj, _ = self.classifier2(image_features_proj)

            return [features, image_features_proj], [cls_scores, cls_scores_proj]

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


clip_model = load_clip_to_cpu('RN50', 18, 9, 16)
clip_model.to("cuda")
label = torch.tensor([0, 1, 2], device='cuda')
prompt = PromptLearner_share(395, clip_model.dtype, clip_model.token_embedding).cuda()
cls_ctx = prompt(label, mode='get_cls_ctx')
prompt_part = PromptLearner_part(clip_model.dtype, clip_model.token_embedding).cuda()
prompts = prompt_part(cls_ctx)
text_encoder = TextEncoder(clip_model).cuda()
# prompts1 = prompts[:, 0, :, :]
# prompts2 = prompts[:, 1, :, :]
text_features = []
for i in range(prompt_part.num_parts):
    print(f'part {i}')
    text_features.append(text_encoder(prompts[i], prompt_part.tokenized_prompts_list[i]))
text_features = torch.stack(text_features, dim=0)  # (num_parts, b, dim)
text_features = text_features.transpose(0, 1)  # (b, num_parts, dim)
print(text_features.shape)
# for i in range(0,7):
#     print(f'part {i}')
#     is_same(prompts1[i], prompts2[i])
# # print(prompts.shape)
