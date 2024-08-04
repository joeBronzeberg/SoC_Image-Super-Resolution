
class Config:

    dataset_name = "BSD100_train"
    val_dataset_name = "BSD100_val"
    test_dataset_name = "test_img"

    hr_shape = (256, 256)
    scale_factor = 4
    n_cpu = 4

    c = 1
    d = 56 #32
    s = 12 #5
    m = 4  #1

    model_name = "FSRCNN" # "FSRCNN-s"
    batch_size = 10
    test_batch_size = 1
    num_epochs = 60

    learning_rate = 0.001
    factor=0.5
    patience=10
    min_learning_rate=1e-6

