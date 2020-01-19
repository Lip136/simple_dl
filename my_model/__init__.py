

def feature_tran(config):
    module = __import__(config["net"]["module_name"])
    func = getattr(module, config["net"]["class_name"])
    return func

