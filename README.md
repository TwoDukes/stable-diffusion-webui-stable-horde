# Stable Horde for Web UI
[Stable Horde](https://stablehorde.net) client for AUTOMATIC1111's [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## Features

- **txt2img**, **img2img**, **depth2img** and **inpaint**
- Sampling methods:
    - **Euler a**
    - **Euler**
    - **LMS** (and **Karras** variant)
    - **Heun**
    - **DPM2** (and **Karras** variant)
    - **DPM2 a** (and **Karras** variant)
    - **DPM++ 2S a** (and **Karras** variant)
    - **DPM++ 2M** (and **Karras** variant)
    - **DPM fast**
    - **DPM adaptive**
- Base models:
    - **stable_diffusion_1.4** (v1.4)
    - **stable_diffusion** (v1.5)
    - **stable_diffusion_inpainting** (v1.5): Generalist model specialized for modifying areas of existing images
    - **stable_diffusion_2.0** (v2.0)
    - **Stable Diffusion 2 Depth** (v2): Generalist model specialized for creating depth maps of existing images, for img2img creations
    - **stable_diffusion_2.1** (v2.1)
- Custom models:
    <!-- [[[cog
    import cog
    import requests

    models = requests.get("https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/db.json")
    models = models.json()
    models_out = []

    for model in models:
        if "config" in models[model]:
            for file in models[model]["config"]["download"]:
                if "file_path" in file:
                    if file["file_path"] == "models/custom":
                        models_out.append("- **{}** (v{}): {}".format(model, models[model]["version"], models[model]["description"]))
                        break

    models_out.sort()
    cog.out("\n".join(models_out))
    ]]] -->
    - **ACertainThing** (v1.0): An improved version of Anything v3 made with ACertainThing, focusing on scenes rather than characters
    - **AIO Pixel Art** (v1): Stable Diffusion fine tuned on pixel art sprites and scenes
    - **Analog Diffusion** (v1.0): A dreambooth model trained on a diverse set of analog photographs
    - **Anything Diffusion** (v3): Highly detailed Anime styled generations
    - **App Icon Diffusion** (v1): Dreambooth model fine tuned on mobile app icons
    - **Arcane Diffusion** (v3): Based on the Arcane TV show
    - **Archer Diffusion** (v1): Based on the Archer's TV show animation style
    - **Asim Simpsons** (v1.0): The Simpsons dreambooth model
    - **Balloon Art** (v1.0): This is the fine-tuned Stable Diffusion model trained on Twisted Balloon images
    - **Borderlands** (v1): Based on Borderlands video game style, trained on characters and scenes
    - **BubblyDubbly** (v1): Dreamy sketched/painted portraits
    - **ChromaV5** (v1.6): generates metalic/chrome looking images
    - **Classic Animation Diffusion** (v1): Popular animation studio classic style generations.
    - **Clazy** (v1): Generates clay-like figures
    - **Comic-Diffusion** (v2): Western Comic book style
    - **Cyberpunk Anime Diffusion** (v1): Cyberpunk anime characters
    - **Dark Victorian Diffusion** (v2.0): finetuned on dark, moody, victorian imagery
    - **Darkest Diffusion** (v1.0): A free and open source Stable Diffusion model created by AI-Characters, trained on the artstyle of the game 'Darkest Dungeon'
    - **Dawgsmix** (v1): anime and realistic anatomy focused merged of anything , trinart , f222 , sd1.5
    - **Double Exposure Diffusion** (v2.0): The Double Exposure Diffusion model, trained specifically on images of people and a few animals
    - **Dreamlike Diffusion** (v1.0): Dreamlike Diffusion 1.0 is SD 1.5 fine tuned on high quality art, made by dreamlike.art
    - **Dreamlike Photoreal** (v1.0): Dreamlike Photoreal 1.0 is a photorealistic Stable Diffusion 1.5 model fine tuned on high quality photos, made by dreamlike.art.
    - **Dungeons and Diffusion** (v1): Generates D&D styled characters, trained on art commissions
    - **Eimis Anime Diffusion** (v1): This model is trained with high quality and detailed anime images
    - **Elden Ring Diffusion** (v2): Based on the Elden Ring video game style
    - **Eternos** (v1.0): A surrealist / Minimalist model
    - **Fantasy Card Diffusion** (v1): fantasy trading card style art, trained on all currently available Magic: the Gathering card art
    - **Funko Diffusion** (v1.0): Stable Diffusion fine tuned on Funko Pop, by PromptHero.
    - **Furry Epoch** (v4): Furry styled generations.
    - **Future Diffusion** (v1.0): This creates high quality 3D images with a futuristic Sci-Fi theme
    - **GTA5 Artwork Diffusion** (v1.0): This model was trained on the loading screens, gta storymode, and gta online DLCs artworks. Which includes characters, background, chop, and some objects. The model can do people and portrait pretty easily, as well as cars, and houses. For some reasons, the model stills automatically include in some game footage, so landscapes tend to look a bit more game-like.
    - **Ghibli Diffusion** (v1): fine-tuned Stable Diffusion model trained on images from Studio Ghibli feature films
    - **Guohua Diffusion** (v1): fine-tuned Stable Diffusion model trained on traditional Chinese paintings
    - **Hassanblend** (v1.4): This model was for creating people
    - **Hentai Diffusion** (v17): Anime focused model with better hands, obscure poses/camera angles and consistent style
    - **Inkpunk Diffusion** (v2): inspired by Gorillaz art, FLCL and Yoji Shinkawa. Trained on images generated from Midjourney
    - **JWST Deep Space Diffusion** (v1): Stable Diffusion fine tuned on JWST imagery
    - **Knollingcase** (v1): generates a glass display case with objects inside, inspired by Sean Preston. Trained on Midjourney images
    - **Mega Merge Diffusion** (v1): SD 1.5 merged with 17 other models
    - **Microscopic** (v1.0): This is the fine-tuned Stable Diffusion model trained on microscopic images
    - **Microworlds** (v1): Isometric microworlds
    - **Midjourney Diffusion** (v1): Stable Diffusion fine tuned on Midjourney v4 images
    - **Midjourney PaintArt** (v1): Midjourney v4 painting style
    - **Min Illust Background** (v1.0): This fine-tuned Stable Diffusion v1.5 model was trained on a selection of artistic works by Sin Jong Hun
    - **Nitro Diffusion** (v1): Multi-Style model trained on Arcane, Archer and Mo-Di
    - **Papercut Diffusion** (v1): Stable Diffusion fine tuned on Paper cut images
    - **Papercutcraft** (v1): Paper Cut Craft is a fine tuned Stable Diffusion model trained on Midjourney images
    - **Poison** (v1): Anything Diffusion fine-tuned to produce high-quality realistic anime styled images
    - **PortraitPlus** (v1.0): This is a dreambooth model trained on a diverse set of close to medium range portraits of people.
    - **RPG** (v1): portraits of charecters in the style of the game Baldur's Gate
    - **Ranma Diffusion** (v1): imitates the style of late '80s early 90's anime, Anything v3 base
    - **Redshift Diffusion** (v1): Dreambooth model trained on high resolution 3D artworks
    - **Robo-Diffusion** (v1): Robot oriented drawing style
    - **Samdoesarts Ultmerge** (v1): Portraits in the style of Sam Yang, merged with chewtoy and orange code's models
    - **Seek.art MEGA** (v1.0): Seek.art MEGA is a general use 'anything' model that significantly improves on 1.5 across dozens of styles. Created by Coreco at seek.art
    - **Smoke Diffusion** (v1.0): This is the fine-tuned Stable Diffusion model trained on images of smoke
    - **Spider-Verse Diffusion** (v1): Based on the Into the Spider-Verse movie's animation style
    - **Squishmallow Diffusion** (v1): Squishmallows
    - **Supermarionation** (v2.0): This is a fine-tuned Stable Diffusion model (based on v1.5) trained on screenshots from Gerry Anderson Supermarionation stop motion animation movie, basically from Thunderbirds tv series
    - **Synthwave** (v1): Stable Diffusion model to create images in Synthwave/outrun style
    - **Trinart Characters** (v2.0): Derrida (formerly TrinArt Characters v2) is a stable diffusion v1-based model that was further improved on the previous characters v1 model. While this is still a versatility and compositional variation anime/manga model like other TrinArt models, when compared to the v1 model, Derrida was focused on more anatomical stability and slightly less on variation due to further multi-epoch training and finetuning.
    - **Tron Legacy Diffusion** (v1): Tron Legacy movie style
    - **Valorant Diffusion** (v1.0): This model was trained on the Valorant agents splash arts, and some extra arts on the official website
    - **Van Gogh Diffusion** (v1): Stable Diffusion model trained on screenshots from the film Loving Vincent, best results with k_euler sampler
    - **Vintedois Diffusion** (v0.1): Vintedois (22h) Diffusion model trained by Predogl and piEsposito with open weights, configs and prompts (as it should be).  This model was trained on a large amount of high quality images with simple prompts to generate beautiful images without a lot of prompt engineering.
    - **Voxel Art Diffusion** (v1): Stable Diffusion fine-tuned on voxel art style
    - **Wavyfusion** (v1): dreambooth model trained on a very diverse dataset ranging from photographs to paintings
    - **Xynthii-Diffusion** (v1): Xynthii-Diffusion (cyclops monster girls)
    - **Yiffy** (v18): Furry styled generations.
    - **Zack3D** (v1): Kink/NSFW oriented furry styled generations.
    - **Zeipher Female Model** (v222): For creating images of nude solo women. Also known as f222
    - **colorbook** (v1): Minimalist coloring book style images
    - **kurzgesagt** (v1): A DreamBooth finetune of Stable Diffusion v1.5 model trained on a bunch of stills from Kurzgesagt videos
    - **mo-di-diffusion** (v1): Popular animation studio modern style generations.
    - **trinart** (v1): Manga styled generations.
    - **vectorartz** (v1): Generate beautiful vector illustration
    - **waifu_diffusion** (v1.3): Anime styled generations.
    <!-- [[[end]]] -->
- Post processing:
    - **CodeFormers** (v0.1.0): Face restoration
    - **GFPGAN** (v1.4): Face restoration
    - **RealESRGAN_x4plus** (v0.1.0): Upscaling

## How to use it

**To prevent loading the local models, add `--ui-debug-mode` to `COMMANDLINE_ARGS`**.

1. If you [registered an account](https://stablehorde.net/register), go to the `Stable Horde Settings` tab and set your `API key`. Leaving the default value will connect anonymously, which is limited.
2. Go to either the `txt2img` or the `img2img` tab and select `Run on Stable Horde` in the `Script` option. Without this option, it will run locally.
3. Set all parameters, both regular and Stable Horde's, and click `Generate`.
