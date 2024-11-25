# !/usr/bin/env python3
"""
Adapt `VGG-Face` as the core model to predict human judgments in the face similarity task.

Run this script via the command line interface (CLI) to train the `VGG-Face` model on human judgments.

!!! tip "What arguments can be passed in CLI?"
    ``` bash
    python -m facesim3d.vgg_predict --help
    ```
"""

# %% Import
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from ut.ils import cprint, send_to_mattermost

from facesim3d.configs import config, paths
from facesim3d.modeling.VGG.models import (
    VGGFaceHumanjudgment,
    VGGFaceHumanjudgmentFrozenCore,
    check_exclusive_gender_trials,
    get_vgg_performance_table,
)
from facesim3d.modeling.VGG.prepare_data import prepare_data_for_human_judgment_model

# %% Set global vars & paths << o >><< o  >><< o  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Set default flags
MAX_EPOCHS: int = 2.0  # low level for testing
SEED: int = 42

# Notification message (Mattermost)
MODEL_MESSAGE: str = """
o-O-.-O-o-.o-O-.-O-o-.o-O-.-O-o-.o-O-.-O-o-.o-O-.-O-o-.o-O-.-O-o
#### Finished model training of '{model_name}'

{hp}

o-O-.-O-o-.o-O-.-O-o-.o-O-.-O-o-.o-O-.-O-o-.o-O-.-O-o-.o-O-.-O-o
"""

# Set logger
logger = logging.getLogger(__name__)  # in configs "__main__"
# set logger name if desired: logger.name = "DESIRED_NAME"

# %% Functions << o  >><< o >><< o  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def evaluate_vgg_face_human_judgment_model(
    model: VGGFaceHumanjudgment,
    data: DataLoader,
    device: str,
    loss_fn: torch.nn.Module | None = None,
    writer: SummaryWriter | None = None,
    global_step: int = 0,
) -> float:
    """Evaluate the `VGG-Face` model for human similarity judgments."""
    model.eval()
    correct = 0
    total = 0

    n_print = np.maximum(len(data) // 25, 10)  # print 25 times over dataset
    running_loss = 0.0
    with torch.no_grad():
        for i, data_i in tqdm(
            enumerate(data),
            desc=f"Iterate through {'test' if loss_fn is None else 'val'} samples",
            total=len(data),
            position=0 if loss_fn is None else 2,
            leave=loss_fn is None,
        ):
            x1, x2, x3, y, _ = data_i.values()  # _ = idx
            outputs = model(x1.to(device), x2.to(device), x3.to(device))
            _, predicted = torch.max(outputs.data, 1)

            total += y.size(0)
            correct += (predicted == y.to(device)).sum().item()

            if loss_fn:
                loss = loss_fn(outputs, y.to(device))
                running_loss += loss.item()
            if (i % n_print) == (n_print - 1):
                if loss_fn:
                    msg = f"Step: {i + 1:6d}, Loss: {loss.item():.5f} | running loss: {running_loss / n_print:.5f}"
                    print(msg)
                if writer is not None:
                    if loss_fn:  # currently only for validation set
                        writer.add_scalar(
                            tag=f"loss/{'test' if loss_fn is None else 'val'}",
                            scalar_value=loss.item(),
                            global_step=i + global_step,
                        )
                    writer.add_scalar(
                        tag=f"running_acc/{'test' if loss_fn is None else 'val'}",
                        scalar_value=correct / total,
                        global_step=i + global_step,
                    )

    acc = correct / total
    msg = f"Accuracy of the network on the {'test' if loss_fn is None else 'validation'} set (n={total}): {acc:.2%}"
    print(msg)
    logger.info(msg)

    return acc


def train_vgg_face_human_judgment_model(
    model: VGGFaceHumanjudgment | VGGFaceHumanjudgmentFrozenCore,
    session: str,
    data_mode: str,
    exclusive_gender_trials: str | None,
    train_data: DataLoader,
    val_data: DataLoader,
    test_data: DataLoader,
    epochs: int,
    device: str,
    learning_rate: float,
    send_message: bool = False,
    seed: int | None = None,
) -> VGGFaceHumanjudgment:
    """Train the `VGG-Face` model for human similarity judgments."""
    # Check arguments
    exclusive_gender_trials = check_exclusive_gender_trials(exclusive_gender_trials=exclusive_gender_trials)
    # TODO: turn "" into e.g. "full_sample", after running all models, & move previous dirs  # noqa: FIX002
    p_fix = "" if exclusive_gender_trials is None else f"{exclusive_gender_trials}_only_trials"  # path_fix

    # Prepare model
    model_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{model.__class__.__name__}"
    model.name = model_name
    save_path = Path(paths.data.models.vggbehave, p_fix, session, f"{model_name}_final.pth")
    save_path_best = Path(str(save_path).replace("_final.pth", "_best.pth"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=str(Path(paths.data.models.vggbehave, p_fix, "runs", session, model_name)))

    # Write input example and graph to tensorboard
    # writer.add_graph(model=model, input_to_model=(img1.to(device), img2.to(device), img3.to(device)),
    #                  verbose=True)

    n_print = np.maximum(len(train_data) // 5, 2)  # print 5 times per epoch
    val_freq = np.maximum(epochs // 10, 2)
    cprint(string=f"\nStart training of '{model_name}' ...\n", col="b", fm="ul")
    best_acc = 0.0  # for validation set, init
    start_time = pd.Timestamp.now()
    epoch_acc = 0.0  # init
    logger.info("Start training of '%s' ...", model_name)
    for epoch in tqdm(range(epochs), desc="Epochs", total=epochs, position=0):
        running_loss = 0.0  # reset for each epoch
        running_corrects = 0
        for i, data in tqdm(
            enumerate(train_data),
            desc="Iterate through training samples",
            total=len(train_data),
            position=1,
            leave=False,
        ):
            x1, x2, x3, y, _ = data.values()  # _ = idx
            optimizer.zero_grad()
            outputs = model(x1.to(device), x2.to(device), x3.to(device))
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, y.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x1.size(0)  # x1.size(0) == train_data.batch_size
            # multiply by batch size to get correct loss since average is taken over batch for x-entropy
            running_corrects += torch.sum(predictions == y.data.to(device))  # OR just y
            if (i % n_print) == (n_print - 1):
                # Save (running) loss to file / use writer
                msg = (
                    f"Epoch: {epoch + 1} | Step: {i + 1:6d} | Current loss: {loss.item():.5f} | "
                    f"Running loss: {running_loss / ((i + 1) * train_data.batch_size):.5f}"
                )
                print(msg, end="\r")
                logger.info(msg)
                writer.add_scalar(tag="loss/train", scalar_value=loss.item(), global_step=i)
                writer.add_scalar(
                    tag="running_acc/train",
                    scalar_value=running_corrects.double() / ((i + 1) * train_data.batch_size),
                    global_step=i * (epoch + 1),
                )

        epoch_loss = running_loss / (len(train_data) * train_data.batch_size)
        # == train_data.sampler.num_samples
        epoch_acc = running_corrects.double() / (len(train_data) * train_data.batch_size)
        cprint(
            string=f"After epoch {epoch + 1}: Training Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2%}", col="g", ts=True
        )

        if (epoch % val_freq) == (val_freq - 1):
            val_acc = evaluate_vgg_face_human_judgment_model(
                model=model,
                data=val_data,
                device=device,
                loss_fn=criterion,
                writer=writer,
                global_step=i * (epoch + 1),
            )
            model.train()  # switch back to train mode

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path_best)
                cprint(
                    string=f"Saved model with current best val accuracy ({best_acc:.1%}) to '{save_path_best}'",
                    col="b",
                    ts=True,
                )

    cprint("\n" + 2 * "*****_" + " FINISHED TRAINING " + "_*****" * 3 + "\n", col="g", ts=True)

    # Last evaluation of model on training and validation set
    accs = {
        "train": epoch_acc.cpu().item(),  # use last epoch's accuracy
        "val": evaluate_vgg_face_human_judgment_model(
            model=model, data=val_data, device=device, loss_fn=criterion, writer=writer
        ),
        "test": evaluate_vgg_face_human_judgment_model(
            model=model, data=test_data, device=device, loss_fn=None, writer=writer
        ),
    }
    for set_name, acc in accs.items():
        msg = f"Final accuracy of the network on the {set_name} set: {acc:.2%}"
        cprint(string=msg, col="g", fm="bo")
        logger.info(msg)

    # Save final model
    if accs["val"] >= best_acc:
        msg = f"Final model has the best val accuracy ({accs['val']:.1%}) and is saved to '{save_path}'"
        torch.save(model.state_dict(), save_path)
        if save_path_best.exists():
            save_path_best.unlink()
    else:
        msg = (
            f"Final model has not the best val accuracy ({accs['val']:.1%}), hence we keep previous best model "
            f"only and rename it to '{save_path}'"
        )
        if save_path_best.exists():
            save_path_best.rename(save_path)
    cprint(string=msg, col="b", ts=True)
    logger.info(msg)

    # Save model hyperparameters to table
    hp_tab = get_vgg_performance_table(hp_search=False, exclusive_gender_trials=exclusive_gender_trials)

    n_heads = len(np.unique(train_data.dataset.dataset.session_data.to_numpy().flatten()))
    # Fill in hyperparameters & accuracies
    hp_tab.loc[len(hp_tab), :] = [
        model_name,
        session,
        data_mode.lower(),
        model.freeze_vgg_core,
        model.last_core_layer,
        model.parallel_bridge,
        model.decision_block_mode,
        train_data.batch_size,
        epochs,
        learning_rate,
        seed,
        device,
        n_heads,
        len(train_data),
        len(val_data),
        (pd.Timestamp.now() - start_time).round(freq="s"),
        accs["train"],
        accs["val"],
        accs["test"],
    ]
    # Convert columns to correct types
    hp_tab.time_taken = hp_tab.time_taken.astype(str)  # writer (below) cannot handle timedelta64
    acc_cols = [c for c in hp_tab.columns if "_acc" in c]
    col_convert = ["bs", "epochs", "seed", "n_train", "n_val"]
    hp_tab[col_convert] = hp_tab[col_convert].astype(int)
    hp_tab[acc_cols] = hp_tab[acc_cols].astype(float).round(3)

    # Also save hyperparameters & accuracies to tensorboard
    writer.add_hparams(
        hparam_dict=hp_tab.loc[len(hp_tab) - 1, hp_tab.columns[:-3]].to_dict(),
        metric_dict=hp_tab.loc[len(hp_tab) - 1, hp_tab.columns[-3:]].to_dict(),
    )

    # Save hp table
    p2_save = (
        paths.data.models.behave.hp_table
        if exclusive_gender_trials is None
        else paths.data.models.behave.hp_table_gender.format(gender=exclusive_gender_trials)
    )
    Path(p2_save).parent.mkdir(parents=True, exist_ok=True)
    hp_tab.to_csv(p2_save, index=False)
    logger.info("Saved hyperparameters & accuracies to '%s'.", p2_save)

    # Close tensorboard writer
    writer.close()

    # Send notification
    if send_message:
        response = send_to_mattermost(
            text=MODEL_MESSAGE.format(
                model_name=model_name, hp=hp_tab.loc[len(hp_tab) - 1].to_markdown()
            ),  # long narrow table (better for Mattermost)
            username=config.PROJECT_NAME,
            incoming_webhook=config.minerva.webhook_in,
            icon_url=config.PROJECT_ICON_URL2,
        )

        if not response.ok:
            msg = f"Could not send message to Mattermost: {response.text}"
            cprint(string=msg, col="r", ts=True)
            logger.error(msg)
    return model


def main():
    """Run the main function of `vgg_predict.py`."""
    # Set device
    if FLAGS.device is None:
        device = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    else:
        device = FLAGS.device
    cprint(string=f"\nUsing {device = }\n", col="y", ts=True)
    logger.info("Using device: %s", device)

    # Set seed
    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

        if device.startswith("cuda"):
            torch.cuda.manual_seed(FLAGS.seed)
            torch.cuda.manual_seed_all(FLAGS.seed)
            torch.backends.cudnn.benchmark = True  # look for optimal algorithms for given train config
        logger.info("Set seed to %s.", FLAGS.seed)

    # Init model
    if FLAGS.freeze_weights:
        # This uses pre-computed activation maps of the VGGFace model
        vgg_hum = (
            VGGFaceHumanjudgmentFrozenCore(
                decision_block=FLAGS.decision_block,
                last_core_layer=FLAGS.last_core_layer,
                parallel_bridge=FLAGS.parallel_bridge,
                session=FLAGS.session,
            )
            .to(device)
            .float()
        )
    else:
        # Here we train end-to-end from image space to human judgments
        vgg_hum = (
            VGGFaceHumanjudgment(
                decision_block=FLAGS.decision_block,
                freeze_vgg_core=FLAGS.freeze_weights,
                last_core_layer=FLAGS.last_core_layer,
                parallel_bridge=FLAGS.parallel_bridge,
                session=FLAGS.session,
            )
            .to(device)
            .float()
        )

    # Train model to predict human judgment
    cprint(
        string=f"\nTraining & testing ({pd.Timestamp.now().ceil(freq='s')})\n"
        f"\t{vgg_hum.__class__.__name__}:\n"
        f"\t\t▸ '{FLAGS.session}' session\n"
        f"\t\t▸ '{FLAGS.decision_block}' decision block\n"
        f"\t\t▸ {'frozen' if FLAGS.freeze_weights else 'unfrozen'} VGG core\n"
        f"\t\t▸ last core layer: '{FLAGS.last_core_layer}'\n"
        f"\t\t▸ parallel bridge: {FLAGS.parallel_bridge}\n"
        f"\t\t▸ learning rate: {FLAGS.learning_rate}\n"
        f"\t\t▸ exclusive gender trials : {FLAGS.exclusive_gender_trials}\n",
        col="b",
        fm="ul",
    )

    # Prepare data
    train_dl, val_dl, test_dl = prepare_data_for_human_judgment_model(
        session=FLAGS.session,
        frozen_core=FLAGS.freeze_weights,
        data_mode=FLAGS.data_mode,
        last_core_layer=FLAGS.last_core_layer,
        split_ratio=(0.7, 0.15, 0.15),  # keep training set large for testing function
        batch_size=FLAGS.batch_size,
        shuffle=FLAGS.shuffle,
        num_workers=FLAGS.num_workers,
        dtype=torch.float32,
        heads=FLAGS.heads,
        size=FLAGS.n_samples,
        exclusive_gender_trials=FLAGS.exclusive_gender_trials,
    )  # keep size small for testing of implementation
    logger.info("Data is prepared for '%s'", vgg_hum.__class__.__name__)

    # Train & test model
    trained_vgg_hum = train_vgg_face_human_judgment_model(
        model=vgg_hum,
        session=FLAGS.session,
        data_mode=FLAGS.data_mode,
        exclusive_gender_trials=FLAGS.exclusive_gender_trials,
        train_data=train_dl,
        val_data=val_dl,
        test_data=test_dl,
        epochs=FLAGS.epochs,
        learning_rate=FLAGS.learning_rate,
        device=device,
        send_message=FLAGS.notification,
        seed=FLAGS.seed,
    )
    logger.info("Finished training of '%s'.", {trained_vgg_hum.__class__.__name__})
    cprint(string=f"\nI, '{trained_vgg_hum.__class__.__name__}', am trained!\n", col="g", fm="bo")


# %% __main__ >><< o >><< o  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Add arg parser
    parser = argparse.ArgumentParser(description="Train expanded VGG-Face on human judgments")  # init

    parser.add_argument("-s", "--session", type=str, action="store", default=None, help="Define session '2D' or '3D'.")

    # Model training parameters
    parser.add_argument(
        "--heads",
        type=str,
        action="store",
        default=None,
        help="Define a subset of heads. Either provide a number of heads or a list of head IDs (e.g., '[1,77,81]'. "
        "If not provided it will take all data.",
    )

    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        action="store",
        default=None,
        help="Define N of subset size. If not provided it will take all data.",
    )

    parser.add_argument(
        "-e", "--epochs", type=int, action="store", default=MAX_EPOCHS, help="Define number of training epochs."
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        action="store",
        default=32,
        help="Define N of total dataset size If not provided it will take all data.",
    )

    parser.add_argument(
        "-l", "--learning_rate", type=float, action="store", default=0.0005, help="Define learning rate."
    )

    parser.add_argument(
        "--notification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send notification after model training is done.",
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        action="store",
        default=None,
        help="Define device to use for training: 'cpu' or 'cuda'.",
    )

    # Model parameters
    parser.add_argument(
        "--decision_block",
        type=str,
        action="store",
        default="conv",
        help="Define architecture of decision block in VGGFaceHumanjudgment* model: 'conv' OR 'fc'",
    )

    parser.add_argument(
        "-f",
        "--freeze_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze weights of VGG core in model for human judgments.",
    )

    # Data parameters
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=True, help="Shuffle data before split."
    )

    parser.add_argument("--num_workers", type=int, action="store", default=0, help="Number of workers for dataloader.")

    parser.add_argument(
        "--data_mode",
        type=str,
        action="store",
        default="3d-reconstructions",
        help="Define data mode of face images for model ['2d-original', '3d-reconstructions', '3d-perspectives'].",
    )

    parser.add_argument(
        "--exclusive_gender_trials",
        type=str,
        action="store",
        default=None,
        help="Indicate if the model should be trained on exclusive gender trials only. "
        "If so, provide the gender: 'female' OR 'male'.",
    )

    parser.add_argument(
        "--last_core_layer",
        type=str,
        action="store",
        default="maxp_5_3",
        help="Define the last layer of VGGFace to keep before the bridge module and decision block.",
    )

    parser.add_argument(
        "--parallel_bridge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the parallel bridge architecture.",
    )

    parser.add_argument(
        "--seed", type=int, action="store", default=SEED, help="Set seed for reproducibility, OR None."
    )

    parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction, default=True, help="Verbose")

    # Parse arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Parse heads (can be None, int, or list of ints)
    if isinstance(FLAGS.heads, str):
        if FLAGS.heads.isdigit():
            FLAGS.heads = int(FLAGS.heads)
        elif FLAGS.heads.startswith("[") or FLAGS.heads.startswith("("):
            FLAGS.heads = [int(h) for h in FLAGS.heads[1:-1].split(",")]

    if FLAGS.verbose:
        cprint(string="\nFLAGS:\n", fm="ul")
        pprint(FLAGS.__dict__)
        cprint(string="\nunparsed:\n", fm="ul")
        print(*unparsed, sep="\n")

    # %% Run main
    logger.info("FLAGS are set. Starting main function ...")
    main()

#  >><< o >><< o  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
