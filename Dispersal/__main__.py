from osgeo import ogr

from sat_preprocess import sat_preprocess
from displace.sat_identify import sat_identify
from sat_post_process import sat_post_process, sat_patch_index, sat_landscape_indicator
from sat_visualize import sat_create, sat_visualize

def main(work_dir):
    sat_preprocess(work_dir)
    sat_identify(work_dir, set_segs=1, start_seg = None)
    sat_post_process(work_dir)
    sat_landscape_indicator(work_dir,indicators=None, time=None)
    sat_patch_index(work_dir, patch_name=None, indices=None)
    sat_create(work_dir)
    sat_visualize(work_dir)
    
if __name__ == "main":
    main()
