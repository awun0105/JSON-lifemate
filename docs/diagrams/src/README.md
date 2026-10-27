# docs/diagrams/src/

Mã nguồn của hai sơ đồ trong README gốc, viết bằng HTML và CSS thường. Sửa file HTML rồi xuất lại PNG bằng Chrome chạy ngầm.

| File | Chức năng |
|---|---|
| `architecture.html` | Sơ đồ kiến trúc tổng quan, xuất ra `../png/readme-architecture.png`. |
| `sample.html` | Sơ đồ một mẫu đi qua hệ thống từng bước, xuất ra `../png/readme-sample-flow.png`. |
| `style.css` | Màu, font, khung ô dùng chung cho hai sơ đồ. |
| `shot.cjs` | Script Node mở file HTML bằng Chrome chạy ngầm và chụp khung `#cv` thành PNG (độ phân giải gấp đôi). Cần `puppeteer-core` và Chrome. |

Xuất lại (từ thư mục gốc của repo):

```bash
node docs/diagrams/src/shot.cjs docs/diagrams/src/architecture.html docs/diagrams/png/readme-architecture.png
node docs/diagrams/src/shot.cjs docs/diagrams/src/sample.html docs/diagrams/png/readme-sample-flow.png
```

`shot.cjs` tìm `puppeteer-core` trong `~/.agents/skills/deck-kit/node_modules`; trên máy khác, cài `npm i puppeteer-core` rồi sửa dòng `require` cho đúng chỗ.
