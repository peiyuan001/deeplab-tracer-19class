class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/peiyuan001/deeplab-tracer+19class/datasets'  # folder that contains VOCdevkit/.
        elif dataset == 'gta5':
            return '/home/peiyuan001/deeplab-tracer+19class/datasets/GTA5/'
            #return 'C:/Users/wenhan002/PycharmProjects/pytorch-deeplab-xception-master/datasets/GTA5/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
