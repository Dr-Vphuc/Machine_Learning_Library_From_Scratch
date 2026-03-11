Dưới đây chúng ta tính các xác suất để một mẫu bất kỳ rơi vào lớp thứ k theo tiên nghiệm P(k) = Nk/N
(tương tự như bài sử dụng Gaussian NB phân loại hoa Iris). Đầu ra đoạn này sẽ là mảng 2 chiều Labels,
trong đó ứng với mỗi phần tử i, giá trị đầu (index 0) sẽ là tên của phân lớp đầu ra; giá trị sau (index 1) sẽ
là tỉ lệ số phần tử (xác suất) của phân lớp đó (P(k) ).

{python}
target_feature = 'cuisine'
Y = df[target_feature]
labels = np.zeros((len(set(Y)), 2), dtype=Y.dtype)
#Liệt kê các nhãn (label) khác nhau (từ trường cuisine) k =1, 2, ... C
#và tính xác suất P_k của mỗi nhãn. Sử dụng nhãn thay cho chỉ số
id = 0
for label in set(Y):
    labels[id, 0] = label
    labels[id, 1] = (Y == label).sum()/len(Y)
    id += 1

Sau đó chúng ta đếm xem mỗi trường có tối đa bao nhiêu lựa chọn từ tập huấn luyện. Chú ý phần này
hoàn toàn có thể thay bằng một mảng cố định, do chúng ta đã biết thông tin về dữ liệu ở phần trên.

def get_distinct_value_in_fields():
    # Get the column names as a Pandas Index object
    column_names_index = df.columns
    tmp_list = column_names_index.values
    num_nomials_per_fields = np.zeros((len(tmp_list ), 2), dtype=tmp_list.dtype)
    # Convert the Index object to a NumPy array
    num_nomials_per_fields[:, 0] = tmp_list
    for i in range(len(tmp_list)):
        X = df[tmp_list[i]]
        unique_elements = set(X)
        # Get the count of unique elements
        num_nomials_per_fields[i, 1] = len(unique_elements)
    return num_nomials_per_fields

num_labels_M = get_distinct_value_in_fields()

Đoạn lệnh tiếp theo tính các xác suất thành phần P(xi | k) theo công thức xấp xỉ (qua Smooth Laplacian)
P(xi | y = k) ~ lambda i k := [|{xm in class k : x_im = xi}| + alpha] / [Nk + M*alpha] . Đầu vào sẽ cần đến số
lựa chọn M của trường thứ i tương ứng (từ đoạn code trước)

def Pxik_feature_per_class(X, xi, num_nomials_M, alpha = 1.0):
    #Returns the prob. of feature per class
    X = np.array(X)
    # count the number of element x_m wich has field i^th : x_im = xi
    count = (X == xi).sum()
    # P(xi| y = k) ~ lambda_ik := [|{xm in class k : x_im = xi}| + alpha] / [N_k + M*alpha]
    return (count + alpha)/(len(X) + num_nomials_M*alpha)


Dự đoán phân lớp y = k đầu ra tương ứng với x – đầu vào, dựa theo công thức
    y = argmaxk=1,...C P(k)P(X | k)

def predict_output_label(X, x_input, p_labels, target_feature):
    num_labels_M = get_distinct_value_in_fields()
    #score for each class
    p = np.log(np.array(p_labels[:, 1], dtype=float))
    for k in range(len(p)):
        #print(X['cuisine']== p_labels[k, 0])
        Xk = X[X[target_feature] == p_labels[k, 0]]
        #print(Xk.shape)
        for i in range(len(x_input)):
            p[k] += np.log(Pxik_feature_per_class(Xk.iloc[:, i+1],x_input.iloc[i], num_labels_M[i+1, 1]))

    y_star = np.argmax(p)
    return p_labels[y_star, 0]