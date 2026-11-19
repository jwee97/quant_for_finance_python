# Running the app: what a machine needs, and where to host it

The dashboard is light. **Training a big network is not**: it needs memory for the feature table and for the network's training state, and CPU time in proportion to the rows, the layers and the epochs. On a small hosted workspace that shows up as a run that **disappears halfway**: the host's memory limit is enforced by the kernel, which kills the process without a word, and the page just stops getting answers.

## What the app now does about it

* **It reads the limits the machine really imposes** (the container's CPU quota and memory limit, not the host's figures) and shows them under *Settings → This machine* (and at `GET /api/system/resources`).
* **It starts as many worker threads as the CPU quota allows**, not one per CPU of the host. A container that shows 32 CPUs and allows one makes numerical libraries fight over it: training is slow and every thread holds its own buffers. A value you set yourself (`OMP_NUM_THREADS`, the model's `num_threads`) is respected.
* **It refuses a run that would not fit, before it starts.** An experiment's memory is estimated (the feature table: assets × days × features, and for a network the training rows, the activations back-propagation keeps and the parameters with their optimiser state) and compared with the memory that is free. The message says what to change. The estimate is rough: it is a guide, not a promise.
* **It stops a run that is about to be killed, and says so.** When the process holds 90% of the memory it may use, the job ends with "Stopped before the machine killed the app…" instead of vanishing. The epochs it had finished stay in the **training history** (*Jobs → Training history*), so you can see how far it got and how it was learning.
* **Training is filed epoch by epoch in the app's database** (`data/user/app.db`), so a crash keeps what happened before it, and a fit the app died in is marked *interrupted* the next time it starts.
* **A backup is one click** (*Data → Storage and backup*): a zip with a consistent copy of every database, made while the app runs. On a host whose disk is wiped on a restart or a redeploy, make one before it happens.

## Making a network smaller

In rough order of how much memory each saves: fewer **assets** and a shorter **history**; fewer **feature groups**; a smaller **`max_rows`** (the number of training rows a network sees); a smaller **`batch_size`**; a smaller **`lookback`**; a smaller **`hidden`** size and fewer **`layers`**. Attention networks (PatchTST, TFT, iTransformer) keep more than recurrent ones. Fewer epochs saves time, not memory. Train one model at a time, and do not run a search (Optuna) and a training together.

## Why Replit is a poor home for the big runs

Replit is a good place to **look at results and run the light parts**; the figures below are what is generally true of its plans and change, so check the current plan pages.

* A workspace and an **Autoscale** deployment have little memory and a fraction of a CPU, and neither is meant for long compute: a long job can be stopped when the app idles or is redeployed.
* On **Autoscale** the disk is not persistent between deployments: anything the app wrote under `data/user` can be gone after a redeploy. A **Reserved VM** keeps its disk but is sized by its plan, and a bigger one costs more.
* Thread counts: see above. The app now caps them, which helps a lot on a fractional CPU, but it cannot create memory.

## Where to run it instead

Pick by what you need. Prices and free tiers move: these are the kinds of place, not an endorsement of a plan.

| You want | Look at | Notes |
|---|---|---|
| **One always-on machine** for the whole app, simplest | A small VPS: Hetzner Cloud, DigitalOcean, Vultr, Linode/Akamai, OVH, AWS Lightsail/EC2, Google Compute Engine | Take **4 vCPU and 8 to 16 GB**, run `docker compose up app` (the repository's `Dockerfile` and `docker-compose.yml`), and keep `./data/user` on the machine's disk or a volume. The memory is yours, the disk persists, nothing idles it. |
| **A free tier that can hold the app** | Oracle Cloud's "Always Free" Arm instances (at the time of writing up to 4 cores and 24 GB in one account) | The most memory for no money, with some friction: capacity in a region is not always available, and PyTorch wheels for Arm must match the Python version. |
| **A managed container, no server to look after** | Fly.io Machines, Railway, Render, Google Cloud Run | Choose an instance with enough **RAM**, and attach a **persistent volume** at `/app/data/user` (Cloud Run has none: keep the databases in a mounted bucket or do not use it for this). Memory is what these charge for. |
| **A GPU for a big network now and then** | Google Colab, Kaggle notebooks, Modal, RunPod, Lambda, Vast.ai, Paperspace | Train there with the command line (`quant ml-train`, see `quant ml-train --help`), then bring the results home: copy the backup zip's databases into `data/user` of the machine that serves the app. Free notebooks give a few hours at a time. |
| **A big machine for a day** | GitHub Codespaces (large machine types, billed by use), or any cloud VM started for the job | The same command-line route; stop it when the run ends. |

A pattern that works well: **serve the dashboard from a small always-on machine, train on a big temporary one**, and move the work between them with the backup (*Data → Storage and backup*, or `GET /api/storage/backups`). The registry, the fundamentals database and the training history are ordinary SQLite files, so restoring is: stop the app, unzip the backup over `data/user`, start it again.

## Checks before a long run

1. *Settings → This machine*: is the memory limit what you expect, and how much is free?
2. Start with a short run (few epochs, a small `max_rows`) and watch the live panel: the **memory** tile shows the process's peak, so you can scale from it.
3. Keep the browser tab open or not: the run goes on without it, and *Jobs* shows how it ended.
