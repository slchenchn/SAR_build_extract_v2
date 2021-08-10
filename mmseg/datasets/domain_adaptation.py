'''
Author: Shuailin Chen
Created Date: 2021-08-04
Last Modified: 2021-08-10
	content: dataset for domain adaptation
'''

import os.path as osp
import mmcv.parallel.collate
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose

@DATASETS.register_module()
class DomainAdaptationDataset(CustomDataset):
    ''' DataSet for domain adapation purpose, offer images of source and target domains simultaneouly
    '''

    def __init__(self,
                 pipeline,
                 img_dir='',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 split={},
                 data_root=None,
                 ann_dir={},
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 domains=('src', 'dst'),
                 domain_map={'src':0, 'dst':1},
                 ):

        self.domains = domains
        self.domain_map = domain_map
        self._check_dict_inputs(img_dir=img_dir, ann_dir=ann_dir,
                                img_suffix=img_suffix,
                                seg_map_suffix=seg_map_suffix, split=split,
                                data_root=data_root)
                                
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        self.pipeline = {}
        self.img_infos = {}
        for domain in self.domains:
            self.pipeline = Compose(pipeline)

            # join paths if data_root is specified
            if self.data_root[domain] is not None:
                if not osp.isabs(self.img_dir[domain]):
                    self.img_dir[domain] = osp.join(self.data_root[domain],
                                                    self.img_dir[domain])
                if not (self.ann_dir[domain] is None 
                        or osp.isabs(self.ann_dir[domain])):
                    self.ann_dir[domain] = osp.join(self.data_root[domain],
                                                    self.ann_dir[domain])
                if not (self.split.get(domain, None) is None 
                        or osp.isabs(self.split[domain])):
                    self.split[domain] = osp.join(self.data_root[domain],
                                                    self.split[domain])

            # load annotation files
            self.img_infos.update({domain:self.load_annotations(self.img_dir[domain],
                                                self.img_suffix[domain],
                                                self.ann_dir[domain],
                                                self.seg_map_suffix[domain], 
                                                self.split.get(domain, None))
            })

    def __len__(self):
        ''' Important, becauset the dataloader need this to determine the data index '''
        return len(self.img_infos[self.domains[0]])

    def _check_dict_inputs(self, **kargs):
        for key, value in kargs.items():
            if isinstance(value, str):
                new_value = {d:value for d in self.domains}
                setattr(self, key, new_value)
            else:
                setattr(self, key, value)

    def get_ann_info(self, idx, domain):
        """Get annotation by index.

        Args:
            idx (int): Index of data.
            domain (str): domain split of data

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[domain][idx]['ann']

    def pre_pipeline(self, results, domain):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir[domain]
        results['seg_prefix'] = self.ann_dir[domain]
        if self.custom_classes:
            results['label_map'] = self.label_map[domain]

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        results = dict(domains=self.domains)
        for domain in self.domains:
            # NOTE: use same index for source and target domain, don't know whether matters or not
            img_info = self.img_infos[domain][idx]
            ann_info = self.get_ann_info(idx, domain=domain)
            results[domain] = dict(img_info=img_info, ann_info=ann_info,
                                    domain=self.domain_map[domain])
            self.pre_pipeline(results[domain], domain=domain)

            results[domain] = self.pipeline(results[domain])
        return results