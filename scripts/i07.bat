python synthesize.py --text "dạ vâng em thấy là mình đang có cái khoản vay thế chấp ở bên vi pi banh nó đang bị thanh toán thiếu mất năm trăm năm mươi nghìn đồng nữa ạ anh xem nó có bổ sung và luôn giúp em được không anh nhở" \
 --restore_step 157000 --mode single -p config/vpb/preprocess.yaml -m config/vpb/model.yaml -t config/vpb/train.yaml --speaker 1


python synthesize.py --text "Em hiểu chị đang gặp khó khăn và chỉ thu xếp thanh toán được trước một phần dư nợ. Tuy nhiên, số tiền này đang chưa đủ để em có thể hỗ trợ thêm. Do đó, để thể hiện thiện chí hợp tác trả nợ và giảm gánh nặng trả lãi, chị thu xếp vay mượn thanh toán tối thiểu ba trăm hai mươi bảy nghìn trong hai ngày tới được không ạ?" \
 --restore_step 200000 --mode single -p config/vpb/preprocess.yaml -m config/vpb/model.yaml -t config/vpb/train.yaml 

 python synthesize.py --restore_step 200000 --mode batch -p config/vpb/preprocess.yaml -m config/vpb/model.yaml -t config/vpb/train.yaml --source 'preprocessed5/speaker1/val.txt'


 python synthesize.py --text "Chào bạn" \
 --restore_step 200000 --mode single -p config/vpb/preprocess.yaml -m config/vpb/model.yaml -t config/vpb/train.yaml 
