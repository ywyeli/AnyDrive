from os.path import isfile, exists
from pathlib import Path
from .NusceneRender import NusceneRender

class Reader:
    def __init__(self,root,
                label_color_map,
                points_filename = 'points.bin',
                ground_filename = 'ground.bin',
                pred_filename = 'pred.bin') -> None:
        self.root = Path(root)
        if not exists(root):
            raise Exception('Data root not found')
        
        self.points_filename = points_filename
        self.ground_filename = ground_filename
        self.pred_filename = pred_filename
        self.label_color_map = label_color_map
        self.cur_index = -1
        self._getdatalen()

    def _valid(self, data_path):
        return exists(data_path) and isfile(data_path / self.ground_filename) \
            and isfile(data_path / self.points_filename) and isfile(data_path / self.pred_filename)

    def _getdatalen(self):
        self.len = 0
        while(self._valid(self.root / f'data{self.len}')): self.len+=1

    def __getitem__(self,idx):
        dir = self.root / f'data{idx}'
        return NusceneRender(dir / self.points_filename, dir / self.ground_filename, dir / self.pred_filename, self.label_color_map)