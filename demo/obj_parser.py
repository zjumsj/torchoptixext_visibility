import numpy as np

def read_file_to_list(file_path):
    """
    Reads a text file into a list, removing trailing newlines and leading whitespace

    Args:
        file_path (str): Path to the input file

    Returns:
        list: Processed lines without newlines or leading whitespace
    """
    lines = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                #processed_line = line.rstrip('\n')
                # Remove trailing newline and leading whitespace
                processed_line = line.rstrip('\n').lstrip()
                lines.append(processed_line)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except Exception as e:
        print(f"Error reading file: {e}")

    return lines

class ObjParser:
    def __init__(self):
        self.lines = []
        self.v = None
        self.vt = None
        self.vn = None

        self.f_v = None
        self.f_vt = None
        self.f_vn = None

    @property
    def f(self):
        return self.f_v

    def process_v_lines(self, lines):
        data = []
        for line in lines:
            tokens = line.split()[1:]
            float_array = np.array([float(token) for token in tokens])
            data.append(float_array)
        if len(data):
            data = np.stack(data,axis=0)
            #print(data.dtype)
        else:
            data = None
        return data

    def parse_f_(self, str):
        tmps = str.split('/')
        if len(tmps) == 1:
            self.f_mode = 0
            return int(tmps[0]), None, None
        elif len(tmps) == 2:
            self.f_mode = 1
            return int(tmps[0]), int(tmps[1]), None
        elif len(tmps) == 3:
            if len(tmps[1]) == 0:
                self.f_mode = 2
                return int(tmps[0]), None, int(tmps[2])
            else:
                self.f_mode = 3
                return int(tmps[0]), int(tmps[1]), int(tmps[2])
        else:
            raise RuntimeError

    def process_f_lines(self, lines):
        f_v = []
        f_vn = []
        f_vt = []
        for line in lines:
            tokens = line.split()[1:]
            assert(len(tokens) > 0)
            v_ = []; vt_ = []; vn_ = []
            o = self.parse_f_(tokens[0])
            token_type = self.f_mode
            if o[0] is not None: v_.append(o[0])
            if o[1] is not None: vt_.append(o[1])
            if o[2] is not None: vn_.append(o[2])
            for token in tokens[1:] :
                o = self.parse_f_(token)
                assert(token_type == self.f_mode)
                if o[0] is not None: v_.append(o[0])
                if o[1] is not None: vt_.append(o[1])
                if o[2] is not None: vn_.append(o[2])
            if len(v_): f_v.append(v_)
            if len(vt_): f_vt.append(vt_)
            if len(vn_): f_vn.append(vn_)
        f_v = np.asarray(f_v,dtype=np.int64) - 1 if len(f_v) else None
        f_vt = np.asarray(f_vt,dtype=np.int64) - 1 if len(f_vt) else None
        f_vn = np.asarray(f_vn,dtype=np.int64) - 1 if len(f_vn) else None
        return f_v, f_vt, f_vn

    def load(self, filename):
        self.lines = read_file_to_list(filename)
        v_lines = []
        vt_lines = []
        vn_lines = []
        f_lines = []
        for line in self.lines:
            if line.startswith('vt '):
                vt_lines.append(line)
            elif line.startswith('vn '):
                vn_lines.append(line)
            elif line.startswith('v '):
                v_lines.append(line)

            elif line.startswith('f '):
                f_lines.append(line)
        self.v = self.process_v_lines(v_lines)
        self.vt = self.process_v_lines(vt_lines)
        self.vn = self.process_v_lines(vn_lines)

        self.f_v, self.f_vt, self.f_n = self.process_f_lines(f_lines)
