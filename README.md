# The San Francisco eXtra Large dataset
Here you can find 3 versions of the San Francisco eXtra Large (SF-XL) dataset:

1) _raw_: there are all 3.43M 360 degrees panoramas of SF-XL, as well as the queries used for the two versions of the test set (namely SF-XL test v1 and SF-XL test v2). There are also crops of the panoramas (for each panorama we obtained 12 partially overlapping crops) inside `raw/train/database`.
You can train CosPlace on `raw/train/database`, and it will automatically sample the needed images.

2) _processed_: there are all 5.6M images that are used to train CosPlace (see Sec. 2.2 of the supplementary) from 8 groups, as well as the val set and the test set.
You can use this to reproduce CosPlace's results without having to download the whole raw dataset, although this offers little flexibility to the training (i.e. these are the samples with M=10, so if you change the value of the hyperparameter M you will get some strange behaviour).

3) _small_: this is a small curated subset of _processed_ which allows you to quickly get started and is only 4.8 GB heavy. Obviously, results won't be as good as when using the _processed_ version, but should be good enough. There is a train, val and test set. The train set is only from 1 group, obtained with _L=12_. To train on this dataset, you should do `$ python train.py --groups_num 1`.
Use this for some quick experiments.

Regarding CosPlace training: you can train it either on `raw/train/database` (see below) or `processed/train`. When training with the default hyperparameters, using one or the other is the same, but if some hyperparameters are changed, using `processed/train` is not suitable anymore, because it only contains the subset of images needed with the default hyperparameters (M, N, alpha and L).


### Available directories

| Directory Path                   | # images | Resolution | Size   |
|----------------------------------|----------|------------|--------|
| raw/train/panoramas              | 3.43M    | 3328 x 512 | 890 GB |
| raw/train/database               | 41.2M    | 512 x 512  |   1 TB |
| processed/train                  | 5.6M     | 512 x 512  | 242 GB |
| processed/val/database           | 8k       | 512 x 512  | 0.4 GB |
| processed/val/queries            | 8k       | 512 x 512  | 0.4 GB |
| processed/test/database          | 2.8M     | 512 x 512  | 122 GB |
| processed/test/queries_v1        | 1000     | -          | 0.1 GB |
| processed/test/queries_v2        | 598      | -          | 0.1 GB |
| processed/test/queries_night     | 466      | -          | 0.1 GB |
| processed/test/queries_occlusion | 76       | -          | 0.1 GB |
| small/train                      | 60k      | 512 x 512  | 2.8 GB |
| small/val/database               | 8k       | 512 x 512  | 0.4 GB |
| small/val/queries                | 8k       | 512 x 512  | 0.4 GB |
| small/test/database              | 27k      | 512 x 512  | 1.3 GB |
| small/test/queries_v1            | 1000     | -          | 0.1 GB |


### Other files

| File Path                                | Note                                                          |
|------------------------------------------|---------------------------------------------------------------|
| README.md                                | README                                                        |
| all_images_paths.txt                     | Paths of all the images within all folders, one path per line |
| metadata.csv                             | Metadata of all panoramas (including altitude, pitch, roll)   |
| processed/train_images_paths.txt         | Paths of all the images within processed/train/               |
| processed/test/database_images_paths.txt | Paths of all the images within processed/test/database/       |
| raw/train/database_images_paths.txt      | Paths of all the images within raw/train/database/            |
| raw/train/panoramas_images_paths.txt     | Paths of all the images within raw/train/panoramas/           |

Files containing the paths can be useful for speeding up the reading of paths, because going through directories with millions of files (e.g. with glob.glob()) can be very slow.


## How to download

To download a directory (or file) you can run this bash command (it requires rsync, does not support scp)
`rsync -rhz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl/REL_PATH .`
for example to download the `processed/test/queries_night` folder you can run
`rsync -rhz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl/processed/test/queries_night .`

You can also download subfolder, or even the whole dataset at once, like this
`rsync -rhz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl .`

You can download a file in the same way, e.g.
`rsync -rhz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl/metadata.csv .`


### Metadata

All available metadata (including altitude, pitch and roll) of all panoramas are contained within the metadata.csv file.
Besides that, metadata (excluding altitude, pitch and roll) is embedded within the name of each image with the following format defined [here](https://github.com/gmberton/VPR-datasets-downloader):
`@ UTM_east @ UTM_north @ UTM_zone_number @ UTM_zone_letter @ latitude @ longitude @ pano_id @ @ heading @ @ @ @ timestamp @ @.jpg`

For example an image can be named: `@0544388.51@4172758.25@10@S@037.70098@-122.49646@kPnABhWJ1kA61eMNPLs1vQ@@210@@@@201606@@.jpg`

Note that depending on the image, some of the fields might be empty (e.g. for the queries_v1 we do not have the heading).

The UTM information are obtained from the latitude and longitude using the `utm` library (install with `pip install utm`). UTM information and latitude-longitude are therefore redundant, and we kept both for simplicity.

For panoramas, heading refers to the center of the image (heading 0 points towards north, 90 is west, 180 south).
For crops (512x512 images extracted from panoramas), heading refers to the leftmost column of the image.


## Visualizations

You can visualize the panoramas directly on Google StreetView using this URL on your browser:
https://www.google.com/maps/@,,3a,75y,,,/data=!3m5!1e1!3m3!1sINSERT_HERE_PANO_ID!2e0!6shttps%3Dmaps_sv.tactile.gps%3D100
For example to visualize the panorama with pano_ID EE53RCj5HDxblYzcSt1Zkw you can browse to:
https://www.google.com/maps/@1,1,3a,75y,90h,100t,0r/data=!3m5!1e1!3m3!1sEE53RCj5HDxblYzcSt1Zkw!2e0!6shttps%3Dmaps_sv.tactile.gps%3D100


## Issues
If you have any question or you think some parts of this README are not clear, feel free to send an email to berton.gabri@gmail.com or gabriele.berton@polito.it


### Cite
If you use the San Francisco eXtra Large please cite
```
@InProceedings{Berton_CVPR_2022_CosPlace,
    author    = {Berton, Gabriele and Masone, Carlo and Caputo, Barbara},
    title     = {Rethinking Visual Geo-Localization for Large-Scale Applications},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4878-4888}
}
```

If you use the queries_night and queries_occlusion please cite
```
@InProceedings{Barbarani_2023_CVPR,
    author    = {Barbarani, Giovanni and Mostafa, Mohamad and Bayramov, Hajali and Trivigno, Gabriele and Berton, Gabriele and Masone, Carlo and Caputo, Barbara},
    title     = {Are Local Features All You Need for Cross-Domain Visual Place Recognition?},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
}
```
