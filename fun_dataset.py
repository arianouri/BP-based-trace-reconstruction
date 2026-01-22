def get_cluster_line_numbers(clusters_file, center_line_number):

    cluster_index = 0  # Start at 0 since the first separator is not a cluster boundary
    first_separator_seen = False
    result_lines = []

    with open(clusters_file, "r") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()

            # Check if line is a separator (one or more "=")
            if stripped != "" and all(ch == "=" for ch in stripped):
                if not first_separator_seen:
                    # First separator only marks the start of the file
                    first_separator_seen = True
                else:
                    # Real separator → next cluster
                    cluster_index += 1
                continue

            # Only start collecting after first separator
            if not first_separator_seen:
                continue

            # If cluster matches target, record the line number
            if cluster_index + 1 == center_line_number:
                if stripped:
                    result_lines.append(lineno)

            # If we have passed target cluster, stop early
            if cluster_index + 1 > center_line_number:
                break

    return result_lines



def get_center_length(centers_file, line_number_center):

    with open(centers_file, "r") as f:
        for lineno, line in enumerate(f, start=1):
            if lineno == line_number_center:
                return len(line.strip())

    raise ValueError("line_number_center exceeds number of lines in file")


def get_pool_size(clusters_file):

    cluster_index = 0  # Start at 0 since the first separator is not a cluster boundary
    first_separator_seen = False

    with open(clusters_file, "r") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()

            # Check if line is a separator (one or more "=")
            if stripped != "" and all(ch == "=" for ch in stripped):
                if not first_separator_seen:
                    # First separator only marks the start of the file
                    first_separator_seen = True
                else:
                    # Real separator → next cluster
                    cluster_index += 1

    return cluster_index