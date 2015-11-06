import codecs
import tables
import pickle
import warnings

def dict2group(f, parent, groupname, dictin, force=False, recursive=True):
    """
    Take a dict, shove it into a PyTables HDF5 file as a new group. Items
    with types that aren't natively supported by PyTables will be serialized
    using pickle and stored as string arrays.

    If 'force == True', any existing child group of the parent node with the
    same name as the new group will be overwritten.

    If 'recursive == True' (default), new groups will be created recursively
    for any items in the dict that are also dicts.
    """

    if force:
        try:
            # overwrite pre-existing group
            if parent._v_pathname.endswith('/'):
                pathstr = parent._v_pathname + groupname
            else:
                pathstr = parent._v_pathname + '/' + groupname
            f.removeNode(pathstr, recursive=True)

        except tables.NoSuchNodeError:
            pass

    g = f.createGroup(parent, groupname)
    f.flush()

    for key, item in dictin.items():

        if isinstance(item, dict):
            if recursive:
                dict2group(f, g, key, item, recursive=recursive,
                           force=force)
            continue

        if force:
            # overwrite any pre-existing arrays at this location
            try:
                f.removeNode(g._v_pathname + '/' + key,
                             recursive=True)
            except tables.NoSuchNodeError:
                pass

        try:
            assert item is not None  # fail gracefully for NoneType
            f.createCArray(g, key, obj=item)
        except (TypeError, ValueError, AssertionError):
            # serialize any types that PyTables can't natively
            # handle
            pickled = codecs.encode(pickle.dumps(item), "base64").decode()
            item = 'OBJ_' + pickled
            f.createArray(g, key, item)
        finally:
            f.flush()

    return g


def group2dict(f, g, recursive=True, warn=True, warn_thresh_nbytes=100E6):
    """
    Traverse a group, pull the contents of its children and return them as
    a Python dictionary, with the node names as the dictionary keys.

    If 'recursive == True' (default), we will recursively traverse child
    groups and put their children into sub-dictionaries, otherwise sub-
    groups will be skipped.

    Since this might potentially result in huge arrays being loaded into
    system memory, the 'warn' option will prompt the user to confirm before
    loading any individual array that is bigger than some threshold (default
    is 100MB)
    """

    def memtest(child, threshold=warn_thresh_nbytes):
        mem = child.size_in_memory
        if mem > threshold:
            print('[!] {0} is {1}MB in size [!]').format(
                    child._v_pathname, mem/1E6
                    )
            confirm = raw_input('Load it anyway? [y/N] >>')
            if confirm.lower() == 'y':
                return True
            else:
                print("Skipping item {0}...").format(g._v_pathname)
        else:
            return True

    outdict = {}
    for child in f.listNodes(g):
        try:
            if isinstance(child, tables.group.Group):
                if recursive:
                    item = group2dict(f, child)
                else:
                    continue
            else:
                if memtest(child):
                    item = child.read()
                    if isinstance(item, str):
                        if item.startswith('OBJ_'):
                            item = item[4:]
                            item = pickle.loads(
                                codecs.decode(item.encode(), "base64"))
                else:
                    continue
            outdict.update({child._v_name: item})
        except tables.NoSuchNodeError:
            warnings.warn(
                'No such node: ' + str(child) + ', skipping...'
                )
            pass
    return outdict
