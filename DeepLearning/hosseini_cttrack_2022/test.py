import threading
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from tqdm import tqdm
import nibabel as nib
from dipy.reconst.shm import sf_to_sh
from dipy.core.sphere import Sphere
from utils.utils import *
from utils.model import *


class NewTest:
    def __init__(
        self,
        n_sig,
        n_tar,
        model_dir,
        data_dir,
        data_list,
        data_file_name,
        b0_name,
        bvec_file_name,
        mask_file_name,
        output_dir=None,
        mul_coe=1,
        output_file_name="wm.nii.gz",
        sh_order=4,
        batch_size=2000,
        num_channels=6,
    ):
        self.n_sig = n_sig
        self.n_tar = n_tar
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.data_list = data_list
        self.data_file_name = data_file_name
        self.b0_name = b0_name
        self.mask_file_name = mask_file_name
        self.bvec_file_name = bvec_file_name
        self.output_dir = output_dir if output_dir is not None else data_dir
        self.mul_coe = mul_coe
        self.output_file_name = output_file_name
        self.sh_order = sh_order
        self.batch_size = batch_size
        self.num_channels = num_channels

    def load_model(self):
        self.model = network(self.num_channels)
        # self.model = network(6)
        self.model.load_weights(self.model_dir)

    def predict_batch(self, dwi, fod, xyz):
        """
        dwi: (batch_size, 3, 3, 3, n_sig)
        fod: (H, W, D, n_tar)
        xyz: (batch_size, 3)
        """
        # y = self.model.predict(dwi, verbose=0)
        y = self.model(dwi, training=False)
        fod[xyz[:, 0], xyz[:, 1], xyz[:, 2], :] = y

    def test(self):
        self.load_model()

        if isinstance(self.data_list, str):
            self.data_list = np.loadtxt(self.data_list, dtype=str)
        self.data_list = [os.path.join(self.data_dir, d) for d in self.data_list]

        for dir in tqdm(self.data_list):
            # if the result file already exists, skip
            if os.path.exists(
                os.path.join(
                    self.output_dir, os.path.basename(dir), self.output_file_name
                )
            ):
                pass  # continue

            dwi_nii = nib.load(os.path.join(dir, self.data_file_name))
            dwi = dwi_nii.get_fdata()

            if self.b0_name is not None:
                b0 = nib.load(os.path.join(dir, self.b0_name)).get_fdata()
                b0 = b0[..., np.newaxis] if b0.ndim == 3 else b0
                dwi = np.divide(dwi, b0, out=np.zeros_like(dwi), where=b0 != 0)
                del b0

            if self.bvec_file_name is not None:
                bvecs = np.loadtxt(os.path.join(dir, self.bvec_file_name)).T
                # dwi = np.clip(dwi, 0, 1)
                sphere_bvecs = Sphere(xyz=bvecs)
                dwi = sf_to_sh(
                    dwi,
                    sphere=sphere_bvecs,
                    sh_order=self.sh_order,
                    basis_type="tournier07",
                )

            mask = (
                nib.load(os.path.join(dir, self.mask_file_name))
                .get_fdata()
                .astype(bool)
            )
            mask = mask[..., 0] if mask.ndim == 4 else mask

            xyz = np.argwhere(mask)
            fod = np.zeros((*mask.shape, self.n_tar))
            dwi_in = np.zeros((len(xyz), 3, 3, 3, self.num_channels))

            for i, (x, y, z) in enumerate(xyz):
                x_slice = slice(max(x - 1, 0), min(x + 2, dwi.shape[0]))
                y_slice = slice(max(y - 1, 0), min(y + 2, dwi.shape[1]))
                z_slice = slice(max(z - 1, 0), min(z + 2, dwi.shape[2]))

                dwi_in[i] = np.pad(
                    dwi[x_slice, y_slice, z_slice],
                    [
                        (0, 3 - x_slice.stop + x_slice.start),
                        (0, 3 - y_slice.stop + y_slice.start),
                        (0, 3 - z_slice.stop + z_slice.start),
                        (0, 0),
                    ],
                    mode="constant",
                )

            # # convert dwi_in to tf tensor, to avoid memory leak
            # dwi_in_tf = tf.convert_to_tensor(dwi_in, dtype=tf.float32)
            # del dwi_in
            # dwi_in = dwi_in_tf
            with tf.device("/cpu:0"):
                dwi_in = tf.convert_to_tensor(dwi_in, dtype=tf.float32)

            fod_out = self.model.predict(
                dwi_in,
                verbose=0,
                batch_size=self.batch_size,
                workers=16,
                use_multiprocessing=True,
            )

            fod[xyz[:, 0], xyz[:, 1], xyz[:, 2], :] = fod_out
            # threads = []
            # for i in range(0, len(xyz), self.batch_size):
            #     batch_xyz = xyz[i:i+self.batch_size] if i + \
            #         self.batch_size < len(xyz) else xyz[i:]
            #     batch_dwi = np.zeros((len(batch_xyz), 3, 3, 3, self.n_sig))
            #     for j, (x, y, z) in enumerate(batch_xyz):
            #         x_slice = slice(max(x - 1, 0), min(x + 2, dwi.shape[0]))
            #         y_slice = slice(max(y - 1, 0), min(y + 2, dwi.shape[1]))
            #         z_slice = slice(max(z - 1, 0), min(z + 2, dwi.shape[2]))

            #         batch_dwi[j] = np.pad(dwi[x_slice, y_slice, z_slice], [(0, 3 - x_slice.stop + x_slice.start), (0,
            #                               3 - y_slice.stop + y_slice.start), (0, 3 - z_slice.stop + z_slice.start), (0, 0)], mode='constant')

            #     t = threading.Thread(
            #         target=self.predict_batch, args=(batch_dwi, fod, batch_xyz))
            #     t.start()
            #     threads.append(t)

            # for t in threads:
            #     t.join()

            # fod /= self.mul_coe
            fod_nii = nib.Nifti1Image(fod, dwi_nii.affine)
            nib.save(
                fod_nii,
                os.path.join(
                    self.output_dir, os.path.basename(dir), self.output_file_name
                ),
            )

            del (
                fod_nii,
                fod,
                dwi_nii,
                dwi,
                mask,
                xyz,
                dwi_in,
                fod_out,
                bvecs,
                sphere_bvecs,
            )


# if __name__ == '__main__':
#     n_sig = 6
#     n_tar = 45
#     model_dir = './CTtrack-MSMT-6.h5'
#
#     data_dir = '/mnt/lrz/data/dHCP_new_new/'
#     data_list = np.loadtxt('/mnt/lrz/data/dHCP_new_new/dhcp_test_list_80.txt', dtype=str)[:]
#     data_file_name = f"dwi_{n_sig}_1000_orig.nii.gz"
#     b0_name = 'b0.nii.gz'
#     mask_file_name = 'mask.nii.gz'
#     bvec_file_name = f"dwi_{n_sig}_1000.bvec"
#     output_dir = '/mnt/lrz/data/dHCP_test_data_new/'
#     mul_coe = 1
#     output_file_name = 'dl_msmt_6_CTtrack.nii.gz'
#     sh_order = 2
#     batch_size = 2000
#
#     # # Use CPU only
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
#     test = NewTest(n_sig, n_tar,  model_dir,
#                    data_dir, data_list, data_file_name, b0_name, bvec_file_name, mask_file_name,
#                    output_dir, mul_coe, output_file_name, sh_order, batch_size)
#     test.test()
#
#     del test


if __name__ == "__main__":
    # data_dir = '/mnt/lrz/data/dHCP_new_new/'
    # data_list = np.loadtxt('/mnt/lrz/data/dHCP_new_new/dhcp_test_list_80.txt', dtype=str)

    data_dir = "/mnt/lrz/data/BCP_sampled/"
    data_list = "/mnt/lrz/data/BCP_sampled/BCP_sampled_test_80.txt"

    scenarios = [
        # (6, 45, 2, 'wm_ss3t_20_0_88_1000.nii.gz', './CTtrack-SS3T-6.h5'),
        (6, 45, 2, "wm.nii.gz", "./CTtrack-MSMT-6.h5", 6),
        # (15, 45, 4, 'wm.nii.gz', './CTtrack-MSMT-15.h5'),
        # (15, 45, 4, 'wm_ss3t_20_0_88_1000.nii.gz', './CTtrack-SS3T-15.h5'),
        # (28, 45, 6, 'wm.nii.gz', './CTtrack-MSMT-28.h5'),
        # (28, 45, 6, 'wm_ss3t_20_0_88_1000.nii.gz', './CTtrack-SS3T-28.h5'),
        # (45, 45, 8, 'wm.nii.gz', './CTtrack-MSMT-45.h5'),
        # (45, 45, 8, 'wm_ss3t_20_0_88_1000.nii.gz', './CTtrack-SS3T-45.h5'),
        # Add more scenarios here
        (12, 45, 2, "wm.nii.gz", "./CTtrack-MSMT-12-early.h5", 6),
        (6, 45, 2, "wm.nii.gz", "./BCP-CTtrack-MSMT-6-early.h5", 6),
        (12, 45, 2, "wm.nii.gz", "./BCP-CTtrack-MSMT-12-early.h5", 6),
    ]

    # data_dir = '/mnt/lrz/data/dHCP_age_new/'
    # data_list = np.loadtxt('/mnt/lrz/data/dHCP_age_new/late_test_55.txt', dtype=str)

    # # get the last 21 subjects
    # data_list = data_list[:20]

    # scenarios = [
    #
    #     # (6, 45, 2, 'wm_ss3t_20_0_88_1000.nii.gz', '/home/lrz/code/CTtrack/CTtrack-SS3T-6-early.h5'),
    #     # (6, 45, 2, 'wm_ss3t_20_0_88_1000.nii.gz', '/home/lrz/code/CTtrack/CTtrack-SS3T-6-late.h5'),
    #     # (6, 45, 2, 'wm.nii.gz', '/home/lrz/code/CTtrack/CTtrack-MSMT-6-early.h5'),
    #     # (6, 45, 2, 'wm.nii.gz', '/home/lrz/code/CTtrack/CTtrack-MSMT-6-late.h5'),
    #     #
    #     # (15, 45, 4, 'wm_ss3t_20_0_88_1000.nii.gz', '/home/lrz/code/CTtrack/CTtrack-SS3T-15-early.h5'),
    #     # (15, 45, 4, 'wm_ss3t_20_0_88_1000.nii.gz', '/home/lrz/code/CTtrack/CTtrack-SS3T-15-late.h5'),
    #     # (15, 45, 4, 'wm.nii.gz', '/home/lrz/code/CTtrack/CTtrack-MSMT-15-early.h5'),
    #     # (15, 45, 4, 'wm.nii.gz', '/home/lrz/code/CTtrack/CTtrack-MSMT-15-late.h5'),
    #     #
    #     # (28, 45, 6, 'wm_ss3t_20_0_88_1000.nii.gz', '/home/lrz/code/CTtrack/CTtrack-SS3T-28-early.h5'),
    #     # (28, 45, 6, 'wm_ss3t_20_0_88_1000.nii.gz', '/home/lrz/code/CTtrack/CTtrack-SS3T-28-late.h5'),
    #     # (28, 45, 6, 'wm.nii.gz', '/home/lrz/code/CTtrack/CTtrack-MSMT-28-early.h5'),
    #     # (28, 45, 6, 'wm.nii.gz', '/home/lrz/code/CTtrack/CTtrack-MSMT-28-late.h5'),
    #
    #     (45, 45, 8, 'wm_ss3t_20_0_88_1000.nii.gz', '/home/lrz/code/CTtrack/CTtrack-SS3T-45-early.h5'),
    #     (45, 45, 8, 'wm_ss3t_20_0_88_1000.nii.gz', '/home/lrz/code/CTtrack/CTtrack-SS3T-45-late.h5'),
    #     (45, 45, 8, 'wm.nii.gz', '/home/lrz/code/CTtrack/CTtrack-MSMT-45-early.h5'),
    #     (45, 45, 8, 'wm.nii.gz', '/home/lrz/code/CTtrack/CTtrack-MSMT-45-late.h5'),
    #
    # ]

    for n_sig, n_tar, sh_order, wm, model_dir, num_channels in scenarios:
        data_file_name = f"dwi_{n_sig}_1000_orig.nii.gz"
        bvec_file_name = f"dwi_{n_sig}_1000.bvec"
        # output_dir = '/mnt/lrz/data/dHCP_test_data_new/'
        output_dir = "/mnt/lrz/data/BCP_test/"
        # output_dir = '/mnt/lrz/data/dHCP_age_test/on_late_data'
        mul_coe = 1
        # output_file_name = f'dl_msmt_{"_BCP__" if "BCP" in model_dir else ""}{n_sig}_CTtrack.nii.gz' if wm == 'wm.nii.gz' else f'dl_ss3t_{n_sig}_CTtrack.nii.gz'
        output_file_name = (
            f'dl_msmt_{"_dHCP__" if "BCP" not in model_dir else ""}{n_sig}_CTtrack.nii.gz'
            if wm == "wm.nii.gz"
            else f"dl_ss3t_{n_sig}_CTtrack.nii.gz"
        )

        # output_file_name = f'dl_{"early" if "early" in model_dir else "late"}_{"msmt" if "MSMT" in model_dir else "ss3t"}_mlp_{n_sig}.nii.gz'
        # output_file_name = f'dl_{"early" if "early" in model_dir else "late"}_{"msmt" if "MSMT" in model_dir else "ss3t"}_CTtrack_{n_sig}.nii.gz'

        batch_size = 1024

        # Instantiate NewTest for current scenario
        test = NewTest(
            n_sig,
            n_tar,
            model_dir,
            data_dir,
            data_list,
            data_file_name,
            "b0.nii.gz",
            bvec_file_name,
            "mask_bet.nii.gz",
            output_dir,
            mul_coe,
            output_file_name,
            sh_order,
            batch_size,
            num_channels,
        )
        test.test()

        del test

    # scenarios = [
    #     (6, 45, 2, 'wm.nii.gz', './BCP-CTtrack-MSMT-6-early.h5', 6),
    #     # (12, 45, 2, 'wm.nii.gz', './BCP-CTtrack-MSMT-12-early.h5', 6),
    # ]
    scenarios = [
        (6, 45, 2, "wm.nii.gz", "./CTtrack-MSMT-6.h5", 6),
        (12, 45, 2, "wm.nii.gz", "./CTtrack-MSMT-12-early.h5", 6),
    ]

    for n_sig, n_tar, sh_order, wm, model_dir, num_channels in scenarios:

        for n_target_subjects in [1, 2, 5, 10]:

            data_file_name = f"dwi_{n_sig}_1000_harmo_{n_target_subjects}.nii.gz"
            bvec_file_name = f"dwi_{n_sig}_1000.bvec"
            # output_dir = '/mnt/lrz/data/dHCP_test_data_new/'
            output_dir = "/mnt/lrz/data/BCP_test/"
            # output_dir = '/mnt/lrz/data/dHCP_age_test/on_late_data'
            mul_coe = 1
            # output_file_name = f'dl_msmt_{"_BCP__" if "BCP" in model_dir else ""}{n_sig}_CTtrack_n_target_{n_target_subjects}.nii.gz' if wm == 'wm.nii.gz' else f'dl_ss3t_{n_sig}_CTtrack.nii.gz'
            output_file_name = (
                f'dl_msmt_{"_dHCP__" if "BCP" not in model_dir else ""}{n_sig}_CTtrack_n_target_{n_target_subjects}.nii.gz'
                if wm == "wm.nii.gz"
                else f"dl_ss3t_{n_sig}_CTtrack.nii.gz"
            )

            # output_file_name = f'dl_{"early" if "early" in model_dir else "late"}_{"msmt" if "MSMT" in model_dir else "ss3t"}_mlp_{n_sig}.nii.gz'
            # output_file_name = f'dl_{"early" if "early" in model_dir else "late"}_{"msmt" if "MSMT" in model_dir else "ss3t"}_CTtrack_{n_sig}.nii.gz'

            batch_size = 1024

            # Instantiate NewTest for current scenario
            test = NewTest(
                n_sig,
                n_tar,
                model_dir,
                data_dir,
                data_list,
                data_file_name,
                "b0.nii.gz",
                bvec_file_name,
                "mask_bet.nii.gz",
                output_dir,
                mul_coe,
                output_file_name,
                sh_order,
                batch_size,
                num_channels,
            )
            test.test()

            del test
