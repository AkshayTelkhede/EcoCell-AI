# Data

The original CSV files remain in the project root to preserve the supplied dataset layout. The pipeline accepts either the root directory or `data/` as its data directory.

- `ECdata.csv`: observed base-station energy readings and the regression target `Energy`.
- `CLdata.csv`: timestamped cell load and energy-saving mode indicators.
- `BSinfo.csv`: static base-station and cell metadata such as radio type, mode, frequency, bandwidth, antenna count, and transmit power.

The source repository does not provide ground truth for the future rows represented by the existing submission files. Therefore, the reproducible evaluation uses a chronological holdout from the labeled `ECdata.csv` rows; the existing submissions are retained as historical artifacts and are not used to claim test performance.
