#!/usr/bin/env python

# Copyright 2016 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# track.py
# Created by Will Gray Roncal on 2016-01-28.
# Email: wgr@jhu.edu

from __future__ import print_function

import numpy as np
import nibabel as nb
import dipy
import dipy.data
import dipy.reconst.dti as dti
from dipy.reconst.dti import TensorModel, fractional_anisotropy, quantize_evecs,color_fa
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier,LocalTracking)
from dipy.direction import DeterministicMaximumDirectionGetter,ProbabilisticDirectionGetter
from dipy.tracking.streamline import Streamlines
from dipy.direction import peaks_from_model
from dipy.tracking.eudx import EuDX
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
from dipy.io.streamline import save_trk
from dipy.viz import window,actor
from dipy.viz.colormap import line_colors

class track():

    def __init__(self):
       # """
       # Tensor and fiber tracking class
       # """
        # WGR:TODO rewrite help text
        pass
    
    def track(self,dwi,gtab,mask):
       # """
       # Tracking with basic tensors and basic eudx - experimental
       # We now force seeding at every voxel in the provided mask for
       # simplicity.  Future functionality will extend these options.
       # **Positional Arguments:**

       #         dwi_file:
       #             - File (registered) to use for tensor/fiber tracking
       #         mask_file:
       #             - Brain mask to keep tensors inside the brain
       #         gtab:
       #             - dipy formatted bval/bvec Structure

       # **Optional Arguments:**
       #         stop_val:
       #             - Value to cutoff fiber track
       # """


        dwi_file = dipy.data.load(dwi)
        data = dwi_file.get_data()

 
        dwi_mask = dipy.data.load(mask)
        mask_file = dwi_mask.get_data()
        #gtab = gradient_table(bval,bvec)

        affine = dwi_file.affine 

        seed_mask = mask_file
        seeds = utils.seeds_from_mask(seed_mask, density=1,affine=affine)
        # use all points in mask
        seedIdx = np.where(mask_file > 0)  # seed everywhere not equal to zero                                     
        seedIdx = np.transpose(seedIdx)
        sphere = get_sphere('symmetric724')
        response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.1)

        csd_model = ConstrainedSphericalDeconvModel(gtab,None,sh_order=6)
        csd_fit = csd_model.fit(data,mask=mask_file)


        tensor_model = dti.TensorModel(gtab)
        ten = tensor_model.fit(data,mask=mask_file)
        FA = fractional_anisotropy(ten.evals)
        classifier = ThresholdTissueClassifier(FA, 0.1)

        dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,max_angle=80.,sphere=default_sphere)


        streamlines_generator = LocalTracking(dg,classifier,seeds,affine,step_size = 0.5)
        #prob_dg= ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,max_angle=80.,sphere=sphere)
        #streamlines_generator = LocalTracking(prob_dg, classifier, seeds, affine,
        #                    step_size=.5, max_cross=1)

        streamlines = Streamlines(streamlines_generator)
        tracks = [e for e in streamlines]
    
        #tensor = csd_fit

        return (ten, tracks)


#if name == "__main__":
#fa = track(dwi="/home/nrajamani/ndmg_demo/sub-0025864/ses-1/dwi/sub-0025864_ses-1_dwi.nii.gz",mask="/home/nrajamani/ndmg_demo/sub-0025864/ses-1/dwi/sub-0025864_ses-1_dwi_bet_mask.nii.gz",bval="/home/nrajamani/ndmg_demo/sub-0025864/ses-1/dwi/sub-0025864_ses-1_dwi.bval",bvec="/home/nrajamani/ndmg_demo/sub-0025864/ses-1/dwi/sub-0025864_ses-1_dwi.bvec")
