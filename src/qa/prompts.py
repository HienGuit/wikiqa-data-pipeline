"""Prompt templates for QA generation and QA judging."""

from __future__ import annotations


EXTRACTION_STANDARD_FEW_SHOT = """
Vi du 1
Context:
Hieu ung nha kinh la qua trinh buc xa nhiet tu be mat hanh tinh duoc hap thu boi cac khi nha kinh trong khi quyen, sau do phat xa lai theo moi huong. Co che nay giup duy tri nhiet do trung binh cua Trai Dat o muc phu hop cho su song. Tuy nhien, khi nong do khi nha kinh tang qua cao do hoat dong cua con nguoi, luong nhiet bi giu lai cung tang theo va lam khi hau am len nhanh hon.
Output:
{"reasoning_log":"Chon thong tin ve vai tro cua hien tuong trong dieu kien tu nhien; answer la exact span ngan gon xuat hien nguyen van trong Context.","question":"Co che nao giup giu nhiet do trung binh cua Trai Dat o muc phu hop cho su song?","answer":"Hieu ung nha kinh"}

Vi du 2
Context:
Bao Yagi do bo vao khu vuc ven bien voi suc gio manh va hoan luu rong. Trong nhieu ngay lien tiep sau do, mua lon keo dai lam nhieu canh dong lua bi ngap ung, cay an qua gay do va he thong thuy loi qua tai. Nganh nong nghiep vi the chiu thiet hai nang ne ca ve nang suat lan chi phi khac phuc sau thien tai.
Output:
{"reasoning_log":"Chon cum nguyen nhan truc tiep gay thiet hai cho nong nghiep; answer la exact span nam nguyen van trong Context.","question":"Nhung yeu to nao duoc neu la nguyen nhan truc tiep khien nong nghiep chiu ton that nang sau bao Yagi?","answer":"mua lon keo dai"}
""".strip()

EXTRACTION_FEW_SHOT = EXTRACTION_STANDARD_FEW_SHOT

EXTRACTION_CONTEXTUAL_FEW_SHOT = """
Vi du 1
Title:
Quy trinh Ung pho Su co Dich vu B2B
Context Above:
Khi xay ra su co dich vu nghiem trong, nhom van hanh phai kich hoat quy trinh phan hoi khan trong vong 30 phut. Sau buoc tiep nhan thong tin ban dau, truong bo phan phu trach khach hang phai goi dien cho doi tac de xac nhan pham vi anh huong va lap bien ban su co tam thoi.
Current Context:
O buoc tiep theo, doanh nghiep thanh lap to ra soat noi bo phoi hop voi phong ky thuat de kiem tra nhat ky he thong, doi chieu cau hinh va xac minh cac diem loi phat sinh trong thoi gian gian doan. Muc tieu cua giai doan nay la tim ra nguyen nhan goc re cua su co trong 24 gio, tu do de xuat bien phap khac phuc va ke hoach phong ngua tai dien.
Output:
{"reasoning_log":"Context Above chi dung de noi sang buoc ke tiep trong quy trinh; answer la exact span neu muc tieu chinh cua giai doan ra soat.","succinct_context":"Trong quy trinh ung pho su co dich vu B2B, doanh nghiep da hoan tat buoc xac nhan pham vi anh huong va lap bien ban ban dau.","question":"Muc tieu chinh cua giai doan phoi hop ra soat giua to noi bo va phong ky thuat la gi?","answer":"tim ra nguyen nhan goc re cua su co"}
""".strip()

EXTRACTION_SYSTEM_PROMPT = """
Ban la chuyen gia tao du lieu doc hieu tieng Viet theo kieu extractive QA.

Muc tieu:
- Viet mot cau hoi tu nhien, tu dung duoc.
- Chon mot answer la exact span ngan gon xuat hien nguyen van trong Current Context.

Kieu trich xuat bat buoc (luan phien su dung de da dang hoa):
- Thuc the (Entity): Nguoi, To chuc, Dia danh.
- Su kien (Event): Dien bien, Cot moc lich su.
- Nguyen nhan/He qua (Causality): Ly do dan den su viec, Tac dong.
- Tinh chat (Property): Dac diem, Trang thai.
- Phuong thuc (Method): Cach thuc, Quy trinh, Giai phap.
- Khai niem (Concept): Dinh nghia, Ten goi chuyen nganh.
- Dinh luong (Quantity/Time): So lieu, Ty le, Thoi gian.

Dau ra:
- Chi tra ve mot object JSON, khong them loi giai thich.
- Standard: reasoning_log, question, answer.
- Contextual: reasoning_log, succinct_context, question, answer.

Nguyen tac:
- Moi mau chi nen hoi mot fact hat nhan; uu tien answer ngan nhat nhung van du nghia.
- Dien dat cau hoi theo so it cho mot muc tieu tra loi; tranh dung "nhung/cac ... nao" neu answer khong phai la mot danh sach ngan.
- Neu answer la so lieu, hay giu phan don vi hoac dinh luong di kem khi can de cau tra loi tu du nghia.
- Question phai duoc suy ra tu Current Context, nhung can dien dat lai tu nhien thay vi chep nguyen cau.
- Khong de question lo nguyen van answer.
- Khong dung cac cach dan chieu nhu "theo doan van", "trong bai nay", "o doan tren".
- Khong chon answer trung han voi Title hoac chi la dai tu/tham chieu mo ho.
- reasoning_log chi can ngan 1-2 cau, neu kieu thong tin duoc chon va xac nhan answer la exact span.
- Neu co succinct_context, no chi dung de noi ngu canh sang Current Context, toi da 2 cau, phai la cau hoan chinh va khong duoc chua answer.
- Neu khong tao duoc mau hop le, tra ve {"error":"insufficient_context"}.
""".strip()

EXTRACTION_USER_TEMPLATE = """
Tao mot mau QA literal.

Few-shot:
{few_shot}

Title:
{title}

Current Context:
\"\"\"
{context}
\"\"\"

Tra ve JSON gom reasoning_log, question, answer.
""".strip()

EXTRACTION_CONTEXTUAL_USER_TEMPLATE = """
Tao mot mau QA literal co contextual prefix.

Few-shot:
{few_shot}

Title:
{title}

Context Above:
\"\"\"
{context_above}
\"\"\"

Current Context:
\"\"\"
{context}
\"\"\"

Context Above giup tao succinct_context va lam ro tham chieu trong question; succinct_context khong duoc chua answer va phai ket thuc thanh mot cau hoan chinh.
Answer bat buoc la exact span chi tu Current Context.
Tra ve JSON gom reasoning_log, succinct_context, question, answer.
""".strip()

MULTI_FEW_SHOT = """
Vi du 1 - Logic
Context:
He thong duong sat do thi so 1 ban dau duoc ky vong hoan thanh vao quy 3 nam nay de kip giam tai cho cac truc giao thong trung tam. Tuy nhien, trong giai doan hoan thien nha ga va he thong tin hieu, du an lien tuc thieu hut vat lieu nhap khau va phai dieu chinh tien do thi cong. Vi vay, ban quan ly quyet dinh doi moc van hanh thuong mai sang dau nam sau thay vi khai thac trong nam nay.
Output:
{"ablation_test_log":"Loai Logic: cau dau neu ke hoach ban dau, cac cau sau neu nguyen nhan va ket qua dieu chinh tien do; can it nhat hai cau moi xac dinh duoc moc thay the.","evidence_span":"He thong duong sat do thi so 1 ban dau duoc ky vong hoan thanh vao quy 3 nam nay de kip giam tai cho cac truc giao thong trung tam. Tuy nhien, trong giai doan hoan thien nha ga va he thong tin hieu, du an lien tuc thieu hut vat lieu nhap khau va phai dieu chinh tien do thi cong. Vi vay, ban quan ly quyet dinh doi moc van hanh thuong mai sang dau nam sau thay vi khai thac trong nam nay.","question":"Sau khi tien do du an bi anh huong boi thieu hut vat lieu, moc van hanh thuong mai nao duoc chon thay cho ke hoach hoan thanh trong quy 3?","answer":"dau nam sau"}

Vi du 2 - So sanh
Context:
Chip Snapdragon 8 Gen 3 duoc toi uu hoa manh cho cac tac vu AI tao sinh, dac biet la xu ly ngon ngu tren thiet bi va tang toc cac mo hinh da phuong thuc co nho. Trong khi do, A17 Pro cua Apple lai nhan manh vao nang luc GPU, ho tro Ray Tracing phan cung va duy tri hieu nang do hoa on dinh cho cac tro choi di dong nang. Hai dong chip vi the duoc quang ba theo hai huong uu tien khac nhau du cung nam trong phan khuc cao cap.
Output:
{"ablation_test_log":"Loai So sanh: cac cau neu hai huong toi uu khac nhau cua hai chip; can doi chieu ca hai moi xac dinh dung doi tuong duoc hoi.","evidence_span":"Chip Snapdragon 8 Gen 3 duoc toi uu hoa manh cho cac tac vu AI tao sinh, dac biet la xu ly ngon ngu tren thiet bi va tang toc cac mo hinh da phuong thuc co nho. Trong khi do, A17 Pro cua Apple lai nhan manh vao nang luc GPU, ho tro Ray Tracing phan cung va duy tri hieu nang do hoa on dinh cho cac tro choi di dong nang.","question":"Neu uu tien mot con chip duoc mo ta la thien ve hieu nang do hoa cho game hon la xu ly ngon ngu AI tren thiet bi, nen chon dong nao?","answer":"A17 Pro cua Apple"}
""".strip()

MULTI_CONTEXTUAL_FEW_SHOT = """
Vi du 1 - Trinh tu
Title:
So tay Y khoa Lam sang
Context Above:
Trong phac do xu ly nhiem trung mau cap tinh tai khoa hoi suc, benh nhan phai duoc danh gia dau hieu sinh ton ngay khi nhap vien. Sau do, nhan vien y te thiet lap duong truyen tinh mach, lay mau nuoi cay va bat dau cac buoc theo doi sat trong giai doan dau.
Current Context:
Sau giai doan xu tri ban dau, benh nhan duoc tiem khang sinh lieu cao de khong che phan ung viem va han che nguy co lan rong nhiem trung. Trong 48 gio tiep theo, bac si tiep tuc theo doi sat cac chi so huyet hoc va dap ung dieu tri. Chi khi so luong bach cau giam xuong muc an toan, e kip moi chuyen sang buoc phau thuat noi soi de loai bo o viem con ton luu.
Output:
{"ablation_test_log":"Loai Trinh tu: Context Above noi sang giai doan dieu tri tiep theo; evidence can ca buoc dung khang sinh va dieu kien mo sang phau thuat.","succinct_context":"Trong phac do dieu tri nhiem trung mau cap tinh, benh nhan da hoan tat giai doan danh gia ban dau, thiet lap duong truyen va lay mau nuoi cay.","evidence_span":"Sau giai doan xu tri ban dau, benh nhan duoc tiem khang sinh lieu cao de khong che phan ung viem va han che nguy co lan rong nhiem trung. Trong 48 gio tiep theo, bac si tiep tuc theo doi sat cac chi so huyet hoc va dap ung dieu tri. Chi khi so luong bach cau giam xuong muc an toan, e kip moi chuyen sang buoc phau thuat noi soi de loai bo o viem con ton luu.","question":"Sau giai doan dung khang sinh lieu cao, benh nhan can dat dieu kien nao truoc khi e kip chuyen sang phau thuat noi soi?","answer":"so luong bach cau giam xuong muc an toan"}
""".strip()

MULTI_SYSTEM_PROMPT = """
Ban la chuyen gia tao du lieu doc hieu tieng Viet cho cau hoi suy luan da cau.

Muc tieu:
- Tao mot cau hoi chi tra loi duoc khi ket hop it nhat 2 cau trong Current Context.
- Chon mot answer la exact span ngan gon xuat hien nguyen van trong Current Context.
- Tra them evidence_span de kiem dinh: do phai la doan nguyen van lien tuc tu Current Context, gom it nhat 2 cau va chua answer.

Kieu suy luan phu hop:
- Logic
- So sanh
- Trinh tu
- Tong hop

Dau ra:
- Chi tra ve mot object JSON.
- Standard: ablation_test_log, evidence_span, question, answer.
- Contextual: ablation_test_log, succinct_context, evidence_span, question, answer.

Nguyen tac:
- Question phai that su can nhieu cau; neu chi mot cau la du thi khong hop le.
- Uu tien cac truong hop ma clue ho tro nam o cau khac voi cau chua answer; neu mot cau da tu tra loi day du thi khong dung.
- Question chi hoi mot y, tu dung duoc, khong lo answer va khong dung cac cach dan chieu nhu "theo doan van".
- Dien dat cau hoi theo so it cho mot muc tieu tra loi; tranh dung "nhung/cac ... nao" neu answer khong phai la mot danh sach ngan.
- Khong tu bia them quan he nhan qua, su dung, so sanh, danh tinh hay lien he ma van ban khong neu ro.
- Answer phai ngan gon, khong trung han voi Title va khong phai la dai tu/tham chieu mo ho.
- Kieu answer phai khop voi cau hoi; vi du cau hoi ve khoang thoi gian khong duoc tra bang mot moc ngay tuyet doi.
- evidence_span phai duoc copy nguyen van tu Current Context, giu tinh lien tuc cua doan goc.
- ablation_test_log chi can ngan 1-2 cau, neu kieu suy luan va vi sao can it nhat hai cau.
- Neu co succinct_context, no chi dung de noi ngu canh sang Current Context, toi da 2 cau, phai la cau hoan chinh va khong duoc chua answer.
- Neu khong tim duoc mau suy luan da cau that su hop le, tra ve {"error":"insufficient_context"}.
""".strip()

MULTI_USER_TEMPLATE = """
Tao mot mau QA inferential.

Few-shot:
{few_shot}

Title:
{title}

Current Context:
\"\"\"
{context}
\"\"\"

Tra ve JSON gom ablation_test_log, evidence_span, question, answer.
""".strip()

MULTI_CONTEXTUAL_USER_TEMPLATE = """
Tao mot mau QA inferential co contextual prefix.

Few-shot:
{few_shot}

Title:
{title}

Context Above:
\"\"\"
{context_above}
\"\"\"

Current Context:
\"\"\"
{context}
\"\"\"

Context Above giup tao succinct_context va lam ro tham chieu; succinct_context khong duoc chua answer va phai ket thuc thanh mot cau hoan chinh.
Evidence_span va answer bat buoc copy chi tu Current Context; khong tom tat hoac noi cac cau cach xa nhau.
Tra ve JSON gom ablation_test_log, succinct_context, evidence_span, question, answer.
""".strip()


UNIFIED_JUDGE_SYSTEM_PROMPT = """
You are a Vietnamese QA judge. Score one QA sample that already passed rule-based span validation.

Use only the given context. The provided reasoning_type is the generator label, not the final truth.
First decide whether the question is literal or inferential:
- literal: one sentence is enough to answer.
- inferential: at least two ideas from the context are needed, or the question asks for a relationship, comparison, sequence, cause/effect, or synthesis.

quality_band:
- weak: awkward, too source-like, vague, or not useful despite being technically valid.
- usable: clear, correct, and good enough for train/eval data.
- strong: natural, well-paraphrased, compact, and representative.

difficulty_band:
- easy: answer is obvious from one nearby sentence.
- medium: requires careful reading, paraphrase, or a direct two-step link.
- hard: requires real synthesis, excluding distractors, or combining several necessary facts.

inferential_validity_band:
- weak: one sentence is enough, or the multi-sentence framing is mostly cosmetic.
- usable: at least two ideas are helpful or needed, but the link is direct.
- strong: multiple sentences are essential; removing one key idea prevents a confident answer.

Notes:
- Literal/extraction samples can still be high quality, but their inferential_validity_band should usually be weak.
- Inferential samples should usually be at least as difficult as comparable literal samples when the reasoning is real.
- Do not reward a sample just because the generator labeled it multi-sentence.

Return exactly one JSON object:
{"detected_reasoning_type":"literal|inferential","quality_band":"weak|usable|strong","difficulty_band":"easy|medium|hard","inferential_validity_band":"weak|usable|strong"}
""".strip()

UNIFIED_JUDGE_FEW_SHOT = """
Example 1
reasoning_type: "extraction"
title: "Doc tau"
succinct_context: ""
ablation_test_log: ""
context: "Trong am nhac, doc tau la hinh thuc bieu dien do mot nguoi duy nhat thuc hien. Khi mot nhac cong doc tau mot nhac cu de the hien mot nhac pham, nguoi ta co the goi la doc tau hoac xolo. Nhung khi mot ca si bieu dien mot bai hat mot minh, thi khong goi la doc tau, ma goi la don ca hoac la linh xuong."
question: "Khi mot ca si bieu dien mot bai hat mot minh, hinh thuc do duoc goi la gi?"
answer: "don ca hoac la linh xuong"
Output: {"detected_reasoning_type":"literal","quality_band":"usable","difficulty_band":"easy","inferential_validity_band":"weak"}

Example 2
reasoning_type: "extraction"
title: "Hieu ung quang dien"
succinct_context: ""
ablation_test_log: ""
context: "Hieu ung quang dien gom hieu ung quang dien ngoai va hieu ung quang dien trong. O hieu ung quang dien ngoai, electron bat ra khoi be mat kim loai. O hieu ung quang dien trong, electron khong bat ra khoi be mat ma tro thanh electron tu do trong long vat dan."
question: "Truong hop cac electron khong bat ra khoi be mat ma tro thanh electron tu do trong long vat dan duoc goi la hieu ung gi?"
answer: "hieu ung quang dien trong"
Output: {"detected_reasoning_type":"literal","quality_band":"usable","difficulty_band":"medium","inferential_validity_band":"weak"}

Example 3
reasoning_type: "multi-sentence"
title: "RNA"
succinct_context: "RNA co cau truc bac cao va co the hoat dong nhu enzyme."
ablation_test_log: "Can ket hop thanh phan nucleotide va tinh chat dien tich."
context: "RNA co cau truc bac cao va co the hoat dong nhu enzyme. Moi nucleotide trong RNA chua mot duong ribose, mot base va mot nhom phosphat. Nhom phosphat tich dien am, khien cho RNA la phan tu mang dien."
question: "Thanh phan nao trong nucleotide khien RNA tro thanh phan tu mang dien?"
answer: "nhom phosphat"
Output: {"detected_reasoning_type":"literal","quality_band":"usable","difficulty_band":"easy","inferential_validity_band":"weak"}

Example 4
reasoning_type: "multi-sentence"
title: "Hon Ngu"
succinct_context: ""
ablation_test_log: "Can ket hop vi tri hanh chinh va cach quan sat toan canh."
context: "Hon Ngu la mot dao o Vinh Bac Bo. Ve hanh chinh dao thuoc pho Nghi Huong thuoc thanh pho Vinh, tinh Nghe An. Muon nhin ro toan canh cua Hon Ngu ta phai dung tu ben song."
question: "De quan sat toan bo Hon Ngu, can dung o ben song thuoc dia danh hanh chinh nao?"
answer: "pho Nghi Huong"
Output: {"detected_reasoning_type":"inferential","quality_band":"usable","difficulty_band":"medium","inferential_validity_band":"usable"}

Example 5
reasoning_type: "multi-sentence"
title: "Opus"
succinct_context: ""
ablation_test_log: "Can ket hop quy uoc so opus voi vi du hai concerto cua Chopin."
context: "Opus la cac nhac pham khi nhac duoc danh so theo trinh tu nhat dinh cua mot tac gia. So opus giup nguoi nghe biet duoc sang tac som hay muon trong su nghiep cua mot nha soan nhac. Frederic Chopin co hai ban concerto cho duong cam; ban op.11 ra doi truoc, con ban op.21 ra doi sau."
question: "Dua vao quy uoc so opus, concerto nao cua Chopin thuoc giai doan sang tac muon hon?"
answer: "op.21"
Output: {"detected_reasoning_type":"inferential","quality_band":"strong","difficulty_band":"hard","inferential_validity_band":"strong"}

Example 6
reasoning_type: "extraction"
title: "He thong tren mot vi mach"
succinct_context: ""
ablation_test_log: ""
context: "He thong tren mot vi mach, hay he thong tren chip, la mot mach tich hop gom nhieu thanh phan cua mot may tinh hoac he thong dien tu tren cung mot chip."
question: "He thong tren mot vi mach con duoc goi la gi?"
answer: "he thong tren chip"
Output: {"detected_reasoning_type":"literal","quality_band":"weak","difficulty_band":"easy","inferential_validity_band":"weak"}
""".strip()

UNIFIED_JUDGE_USER_TEMPLATE = """
Judge this QA sample.

reasoning_type: "{reasoning_type}"
title: "{title}"
succinct_context: "{succinct_context}"
ablation_test_log: "{ablation_test_log}"
context: "{context}"
question: "{question}"
answer: "{answer}"

Return exactly one JSON:
{{"detected_reasoning_type":"literal|inferential","quality_band":"weak|usable|strong","difficulty_band":"easy|medium|hard","inferential_validity_band":"weak|usable|strong"}}
""".strip()
