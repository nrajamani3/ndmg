#!/usr/bin/env python

# Copyright 2014 Open Connectome Project (http://openconnecto.me)
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

from itertools import combinations
import numpy as np
import nibabel as nb
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.direction import peaks_from_model
from dipy.tracking.eudx import EuDX
from dipy.data import get_sphere


class track(object):
    def __init__(self, attr=None):
        """
        Initializes the graph with nodes corresponding to the number of ROIs

        **Positional Arguments:**

                N:
                    - Number of rois
                rois:
                    - Set of ROIs as either an array or niftii file)
                attr:
                    - Node or graph attributes. Can be a list. If 1 dimensional
                      will be interpretted as a graph attribute. If N
                      dimensional will be interpretted as node attributes. If
                      it is any other dimensional, it will be ignored.
        """
        # WGR:TODO rewrite help text
        pass

    def compute_tensors(self, dti_vol, atlas_file, gtab, attr=None):
        # WGR:TODO figure out how to organize tensor options and formats
        # WGR:TODO figure out how to deal with files on disk vs. in workspace

        """
        Takes registered DTI image and produces tensors

        **Positional Arguments:**

                dti_vol:
                    - Registered DTI volume, from workspace.
                atlas_file:
                    - File containing an atlas (or brain mask).
                gtab:
                    - Structure containing dipy formatted bval/bvec information
        """

        labeldata = nib.load(atlas_file)

        label = labeldata.get_data()

        """
        Create a brain mask. Here we just threshold labels.
        """

        mask = (label > 0)

        gtab.info
        print data.shape
        """
        For the constrained spherical deconvolution we need to estimate the
        response function (see :ref:`example_reconst_csd`) and create a model.
        """

        response, ratio = auto_response(gtab, dti_vol, roi_radius=10,
                                        fa_thr=0.7)

        csd_model = ConstrainedSphericalDeconvModel(gtab, response)

        """
        Next, we use ``peaks_from_model`` to fit the data and calculated
        the fiber directions in all voxels.
        """

        sphere = get_sphere('symmetric724')

        csd_peaks = peaks_from_model(model=csd_model,
                                     data=data,
                                     sphere=sphere,
                                     mask=mask,
                                     relative_peak_threshold=.5,
                                     min_separation_angle=25,
                                     parallel=True)

        """
        For the tracking part, we will use ``csd_model`` fiber directions
        but stop tracking where fractional anisotropy (FA) is low (< 0.1).
        To derive the FA, used as a stopping criterion, we need to fit a
        tensor model first. Here, we use weighted least squares (WLS).
        """
        print 'tensors...'

        tensor_model = TensorModel(gtab, fit_method='WLS')
        tensor_fit = tensor_model.fit(data, mask)

        FA = fractional_anisotropy(tensor_fit.evals)

        """
        In order for the stopping values to be used with our tracking
        algorithm we need to have the same dimensions as the
        ``csd_peaks.peak_values``. For this reason, we can assign the
        same FA value to every peak direction in the same voxel in
        the following way.
        """

        stopping_values = np.zeros(csd_peaks.peak_values.shape)
        stopping_values[:] = FA[..., None]
        print datetime.now() - startTime

        pass

    def fibers_eudx(self, stopping_values, peak_indices, seeds,
                    odf_vertices, cutoff, attr=None):
        # WGR:TODO figure out how to organize tensor options and formats
        # WGR:TODO figure out how to deal with files on disk vs. in workspace

        """
        Tensors are used to construct fibers

        **Positional Arguments:**

                dti_vol:
                    - Registered DTI volume, from workspace.
                atlas_file:
                    - File containing an atlas (or brain mask).
                gtab:
                    - Structure containing dipy formatted bval/bvec information
        """

        streamline_generator = EuDX(stopping_values,
                                    csd_peaks.peak_indices,
                                    seeds=10**6,
                                    odf_vertices=sphere.vertices,
                                    a_low=0.1)

        streamlines = [streamline for streamline in streamline_generator]

        return streamlines

        pass