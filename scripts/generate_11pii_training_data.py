#!/usr/bin/env python3
"""
生成11种新PII的训练数据
基于零样本测试结果优化的数据生成策略
"""
import json
import random
import re
from typing import List, Dict, Tuple
from datetime import datetime

# 中国省份简称（用于车牌号）
PROVINCES = [
    '京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘', '皖', '鲁',
    '新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', '蒙', '陕', '吉', '闽',
    '贵', '粤', '青', '藏', '川', '宁', '琼'
]

# 常见中文姓氏
SURNAMES = ['王', '李', '张', '刘', '陈', '杨', '黄', '赵', '周', '吴']

# 常见银行名称
BANKS = ['工商银行', '建设银行', '农业银行', '中国银行', '交通银行', '招商银行', '浦发银行', '民生银行']

class PIIDataGenerator:
    def __init__(self, output_file: str = "data/generated_11pii_training.jsonl"):
        self.output_file = output_file
        self.generated_samples = []

    def luhn_checksum(self, card_number: str) -> int:
        """Luhn算法计算校验和"""
        def digits_of(n):
            return [int(d) for d in str(n)]
        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d*2))
        return checksum % 10

    def generate_bank_card(self) -> str:
        """生成符合Luhn算法的银行卡号"""
        # 银行卡前6位BIN码（常见银行）
        bins = ['621700', '622202', '622208', '621226', '622700', '622848']
        bin_code = random.choice(bins)

        # 16位或19位
        if random.random() < 0.5:
            # 16位卡号
            length = 16
        else:
            # 19位卡号
            length = 19

        # 生成卡号主体（不含校验位）
        card_body = bin_code + ''.join([str(random.randint(0, 9)) for _ in range(length - len(bin_code) - 1)])

        # 计算Luhn校验位
        checksum = self.luhn_checksum(card_body + '0')
        check_digit = (10 - checksum) % 10

        return card_body + str(check_digit)

    def generate_vehicle_plate(self) -> str:
        """生成中国车牌号"""
        province = random.choice(PROVINCES)
        letter = random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')

        # 50% 新能源车牌（6位），50% 普通车牌（5位）
        if random.random() < 0.5:
            # 新能源：数字+字母混合
            numbers = ''.join([random.choice('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(5)])
        else:
            # 普通：5位数字/字母
            numbers = ''.join([random.choice('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(4)])

        return f"{province}{letter}{numbers}"

    def generate_ip_address(self) -> str:
        """生成IPv4地址"""
        # 常见内网IP段
        if random.random() < 0.3:
            return f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
        elif random.random() < 0.5:
            return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        elif random.random() < 0.7:
            return f"172.{random.randint(16,31)}.{random.randint(0,255)}.{random.randint(1,254)}"
        else:
            # 公网IP
            return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

    def generate_ipv6_address(self) -> str:
        """生成IPv6地址"""
        # 简化格式和完整格式混合
        if random.random() < 0.3:
            # 简化格式（::）
            return f"2001:db8::{random.randint(1,65535):x}"
        else:
            # 完整格式
            segments = [f"{random.randint(0,65535):04x}" for _ in range(8)]
            return ':'.join(segments)

    def generate_postal_code(self) -> str:
        """生成邮政编码"""
        # 常见城市邮编范围
        ranges = [
            (100000, 102800),  # 北京
            (200000, 202100),  # 上海
            (510000, 511700),  # 广州
            (518000, 518133),  # 深圳
            (310000, 312000),  # 杭州
            (210000, 211200),  # 南京
        ]
        start, end = random.choice(ranges)
        return str(random.randint(start, end))

    def generate_unified_social_credit_code(self) -> str:
        """生成统一社会信用代码（18位）"""
        # 格式：登记管理部门代码(1) + 机构类别代码(1) + 登记管理机关行政区划码(6) + 主体标识码(9) + 校验码(1)
        dept_codes = ['1', '5', '9', 'Y']  # 1-机构编制, 5-民政, 9-工商, Y-其他
        org_codes = ['1', '2', '3', '9']   # 1-企业, 2-个体工商户, 3-农民专业合作社, 9-其他

        code = random.choice(dept_codes) + random.choice(org_codes)
        code += str(random.randint(110000, 659001))  # 行政区划码（6位）
        code += ''.join([random.choice('0123456789ABCDEFGHJKLMNPQRTUWXY') for _ in range(9)])

        # 校验码（简化处理）
        code += random.choice('0123456789ABCDEFGHJKLMNPQRTUWXY')

        return code

    def generate_passport(self) -> str:
        """生成中国护照号"""
        prefix = random.choice(['E', 'G', 'P'])  # E-普通护照, G-公务护照, P-外交护照
        number = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        return f"{prefix}{number}"

    def generate_hk_macau_pass(self) -> str:
        """生成港澳通行证号"""
        prefix = random.choice(['C', 'H'])  # C-往来港澳通行证, H-往来香港通行证
        length = random.choice([8, 10])  # 8位或10位
        number = ''.join([str(random.randint(0, 9)) for _ in range(length)])
        return f"{prefix}{number}"

    def generate_driver_license(self) -> str:
        """生成驾驶证号（同身份证格式）"""
        # 行政区划码（6位）
        area_codes = ['110101', '310106', '440106', '500106', '330106', '320106']
        area = random.choice(area_codes)

        # 出生日期（8位）
        year = random.randint(1970, 2005)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        birth = f"{year}{month:02d}{day:02d}"

        # 顺序码（3位）
        sequence = f"{random.randint(1, 999):03d}"

        # 校验码
        check_digit = random.choice('0123456789X')

        return f"{area}{birth}{sequence}{check_digit}"

    def generate_social_security_card(self) -> str:
        """生成社保卡号（同身份证格式或独立编号）"""
        if random.random() < 0.7:
            # 70%使用身份证格式
            return self.generate_driver_license()
        else:
            # 30%使用独立编号（16位）
            return ''.join([str(random.randint(0, 9)) for _ in range(16)])

    def generate_mac_address(self) -> str:
        """生成MAC地址"""
        # 随机选择分隔符
        separator = random.choice([':', '-'])
        octets = [f"{random.randint(0, 255):02X}" for _ in range(6)]
        return separator.join(octets)

    # ========== 数据生成函数（每种PII类型） ==========

    def generate_bank_card_samples(self, count: int = 2000):
        """生成银行卡号样本"""
        print(f"\n🔄 生成银行卡号样本 ({count}个)...")

        contexts = [
            "请将款项转至银行卡{value}，户名{name}。",
            "我的工资卡号是{value}，开户行是{bank}。",
            "银行卡号{value}已激活，请核实。",
            "转账至{value}，备注：工资。",
            "储蓄卡{value}余额不足，请充值。",
            "信用卡{value}已还款，账单结清。",
            "请绑定银行卡{value}进行支付。",
            "卡号{value}在ATM机取现500元。",
            "{bank}卡号{value}，密码已重置。",
            "持卡人姓名{name}，卡号{value}。",
        ]

        for i in range(count):
            card_number = self.generate_bank_card()
            context = random.choice(contexts)
            name = random.choice(SURNAMES) + random.choice(['明', '红', '强', '伟', '芳', '丽'])
            bank = random.choice(BANKS)

            text = context.format(value=card_number, name=name, bank=bank)

            # 提取所有实体
            entities = [{"type": "BANK_CARD", "value": card_number}]
            if '{name}' in context:
                entities.append({"type": "PERSON_NAME", "value": name})
            if '{bank}' in context:
                entities.append({"type": "ORGANIZATION", "value": bank})

            self.generated_samples.append({
                "input": text,
                "output": {"entities": entities}
            })

        print(f"✅ 银行卡号样本生成完成")

    def generate_vehicle_plate_samples(self, count: int = 1500):
        """生成车牌号样本"""
        print(f"\n🔄 生成车牌号样本 ({count}个)...")

        contexts = [
            "车牌号{value}的车辆违章了，请尽快处理。",
            "停车场监控显示，{value}号车在18:30离开。",
            "车牌{value}已登记，车主为{name}。",
            "请{value}车主移车，挡住消防通道了。",
            "{value}在高速公路超速行驶，罚款200元。",
            "车辆{value}年检到期，请及时办理。",
            "ETC卡已绑定车牌{value}。",
            "号牌为{value}的车辆涉嫌套牌。",
            "车牌{value}在事故现场被拍到。",
            "请提供车牌号{value}的行驶证。",
        ]

        for i in range(count):
            plate = self.generate_vehicle_plate()
            context = random.choice(contexts)
            name = random.choice(SURNAMES) + random.choice(['先生', '女士', '师傅'])

            text = context.format(value=plate, name=name)

            entities = [{"type": "VEHICLE_PLATE", "value": plate}]
            if '{name}' in context:
                entities.append({"type": "PERSON_NAME", "value": name})

            self.generated_samples.append({
                "input": text,
                "output": {"entities": entities}
            })

        print(f"✅ 车牌号样本生成完成")

    def generate_ip_address_samples(self, count: int = 500):
        """生成IP地址样本"""
        print(f"\n🔄 生成IP地址样本 ({count}个)...")

        contexts = [
            "服务器IP地址{value}出现异常，请检查。",
            "登录记录显示，来自{value}的访问被拒绝。",
            "请将防火墙规则添加IP{value}。",
            "数据库连接地址为{value}，端口3306。",
            "API接口地址：http://{value}:8080/api",
            "网关IP为{value}，请配置静态路由。",
            "DHCP分配了IP地址{value}给客户端。",
            "监控显示{value}流量异常，疑似攻击。",
            "VPN连接成功，分配IP{value}。",
            "ping {value}超时，网络不通。",
        ]

        for i in range(count):
            ip = self.generate_ip_address()
            context = random.choice(contexts)
            text = context.format(value=ip)

            self.generated_samples.append({
                "input": text,
                "output": {"entities": [{"type": "IP_ADDRESS", "value": ip}]}
            })

        print(f"✅ IP地址样本生成完成")

    def generate_ipv6_address_samples(self, count: int = 500):
        """生成IPv6地址样本"""
        print(f"\n🔄 生成IPv6地址样本 ({count}个)...")

        contexts = [
            "IPv6地址{value}已分配。",
            "服务器IPv6为{value}，请配置路由。",
            "双栈网络环境下，IPv6地址为{value}。",
            "请将DNS解析到IPv6地址{value}。",
            "隧道接口IPv6地址：{value}",
            "客户端获取IPv6地址{value}失败。",
            "测试IPv6连通性：ping6 {value}",
            "CDN节点IPv6地址为{value}。",
            "SLAAC自动配置IPv6地址{value}。",
            "防火墙规则允许IPv6地址{value}访问。",
        ]

        for i in range(count):
            ipv6 = self.generate_ipv6_address()
            context = random.choice(contexts)
            text = context.format(value=ipv6)

            self.generated_samples.append({
                "input": text,
                "output": {"entities": [{"type": "IPV6_ADDRESS", "value": ipv6}]}
            })

        print(f"✅ IPv6地址样本生成完成")

    def generate_postal_code_samples(self, count: int = 800):
        """生成邮政编码样本"""
        print(f"\n🔄 生成邮政编码样本 ({count}个)...")

        contexts = [
            "收件地址：{address}，邮编{value}。",
            "请将发票寄至{address}，邮政编码{value}。",
            "通讯地址：{address}，邮编：{value}",
            "邮寄地址填写{address}，ZIP码{value}。",
            "{address}，邮政编码：{value}，收件人{name}。",
            "公司地址：{address}，邮编{value}。",
            "户籍地址：{address}，邮编{value}。",
            "快递单号查询，邮编{value}，地址{address}。",
            "请确认邮编{value}对应地址{address}。",
            "发货地址：{address}（邮编{value}）",
        ]

        addresses = [
            "北京市朝阳区建国路1号",
            "上海市浦东新区陆家嘴环路1000号",
            "广州市天河区天河路123号",
            "深圳市南山区科技园南区",
            "杭州市西湖区文三路456号",
            "南京市鼓楼区中山路789号",
        ]

        for i in range(count):
            postal = self.generate_postal_code()
            context = random.choice(contexts)
            address = random.choice(addresses)
            name = random.choice(SURNAMES) + random.choice(['明', '红', '强'])

            text = context.format(value=postal, address=address, name=name)

            entities = [
                {"type": "ADDRESS", "value": address},
                {"type": "POSTAL_CODE", "value": postal}
            ]
            if '{name}' in context:
                entities.append({"type": "PERSON_NAME", "value": name})

            self.generated_samples.append({
                "input": text,
                "output": {"entities": entities}
            })

        print(f"✅ 邮政编码样本生成完成")

    def generate_unified_social_credit_code_samples(self, count: int = 1500):
        """生成统一社会信用代码样本"""
        print(f"\n🔄 生成统一社会信用代码样本 ({count}个)...")

        contexts = [
            "公司统一社会信用代码：{value}，请核实。",
            "企业信用代码{value}已通过工商局验证。",
            "{company}的社会信用代码为{value}。",
            "营业执照号（统一社会信用代码）：{value}",
            "请提供贵司统一信用代码{value}的证明文件。",
            "税务登记使用统一社会信用代码{value}。",
            "企业{company}，信用代码{value}。",
            "招标要求提供统一社会信用代码，我司为{value}。",
            "合同甲方：{company}，信用代码{value}。",
            "统一社会信用代码{value}已列入经营异常名录。",
        ]

        companies = [
            "北京科技有限公司",
            "上海贸易有限公司",
            "深圳软件科技有限公司",
            "广州工业集团",
            "杭州信息技术有限公司",
        ]

        for i in range(count):
            code = self.generate_unified_social_credit_code()
            context = random.choice(contexts)
            company = random.choice(companies)

            text = context.format(value=code, company=company)

            entities = [{"type": "UNIFIED_SOCIAL_CREDIT_CODE", "value": code}]
            if '{company}' in context:
                entities.append({"type": "ORGANIZATION", "value": company})

            self.generated_samples.append({
                "input": text,
                "output": {"entities": entities}
            })

        print(f"✅ 统一社会信用代码样本生成完成")

    def generate_passport_samples(self, count: int = 1500):
        """生成护照号样本"""
        print(f"\n🔄 生成护照号样本 ({count}个)...")

        contexts = [
            "我的护照号是{value}，有效期至{year}年。",
            "请出示护照{value}办理登机手续。",
            "旅客{name}，护照号码{value}。",
            "签证申请需要护照号{value}。",
            "护照{value}已过期，需重新办理。",
            "出入境记录显示，护照号{value}于{date}入境。",
            "请提供护照号{value}的复印件。",
            "{name}的护照号为{value}，国籍中国。",
            "酒店登记：护照{value}，姓名{name}。",
            "海关申报单，护照号码{value}。",
        ]

        for i in range(count):
            passport = self.generate_passport()
            context = random.choice(contexts)
            name = random.choice(SURNAMES) + random.choice(['明', '红', '强', '芳'])
            year = random.randint(2025, 2035)
            date = f"2024年{random.randint(1,12)}月{random.randint(1,28)}日"

            text = context.format(value=passport, name=name, year=year, date=date)

            entities = [{"type": "PASSPORT", "value": passport}]
            if '{name}' in context:
                entities.append({"type": "PERSON_NAME", "value": name})

            self.generated_samples.append({
                "input": text,
                "output": {"entities": entities}
            })

        print(f"✅ 护照号样本生成完成")

    def generate_hk_macau_pass_samples(self, count: int = 1500):
        """生成港澳通行证样本"""
        print(f"\n🔄 生成港澳通行证样本 ({count}个)...")

        contexts = [
            "港澳通行证{value}已过期，需重新办理。",
            "请携带港澳通行证{value}通关。",
            "{name}的往来港澳通行证号为{value}。",
            "通行证{value}有效期内可多次往返。",
            "请提供港澳通行证{value}办理签注。",
            "往来港澳通行证号码：{value}，持证人{name}。",
            "香港入境处要求提供通行证号{value}。",
            "港澳通行证{value}签注已用完。",
            "{name}持通行证{value}前往香港旅游。",
            "通行证{value}在罗湖口岸通关。",
        ]

        for i in range(count):
            hk_pass = self.generate_hk_macau_pass()
            context = random.choice(contexts)
            name = random.choice(SURNAMES) + random.choice(['明', '红', '强', '芳', '伟'])

            text = context.format(value=hk_pass, name=name)

            entities = [{"type": "HK_MACAU_PASS", "value": hk_pass}]
            if '{name}' in context:
                entities.append({"type": "PERSON_NAME", "value": name})

            self.generated_samples.append({
                "input": text,
                "output": {"entities": entities}
            })

        print(f"✅ 港澳通行证样本生成完成")

    def generate_driver_license_samples(self, count: int = 1500):
        """生成驾驶证号样本"""
        print(f"\n🔄 生成驾驶证号样本 ({count}个)...")

        contexts = [
            "驾驶证号{value}，准驾车型C1。",
            "请提供驾驶证{value}进行核查。",
            "{name}的驾驶证号码为{value}。",
            "驾照{value}扣分已达11分，请注意。",
            "驾驶证{value}年审到期，请及时办理。",
            "交警查验驾驶证号{value}，证件有效。",
            "驾驶人{name}，证号{value}，准驾C1D。",
            "请出示驾驶证{value}和行驶证。",
            "驾驶证{value}因违章被暂扣。",
            "{name}持驾驶证{value}驾驶营运车辆。",
        ]

        for i in range(count):
            license_no = self.generate_driver_license()
            context = random.choice(contexts)
            name = random.choice(SURNAMES) + random.choice(['明', '强', '伟', '军', '峰'])

            text = context.format(value=license_no, name=name)

            entities = [{"type": "DRIVER_LICENSE", "value": license_no}]
            if '{name}' in context:
                entities.append({"type": "PERSON_NAME", "value": name})

            self.generated_samples.append({
                "input": text,
                "output": {"entities": entities}
            })

        print(f"✅ 驾驶证号样本生成完成")

    def generate_social_security_card_samples(self, count: int = 1500):
        """生成社保卡号样本"""
        print(f"\n🔄 生成社保卡号样本 ({count}个)...")

        contexts = [
            "社保卡号{value}，单位缴费正常。",
            "请提供社会保障卡号{value}查询。",
            "{name}的社保卡号为{value}。",
            "社保卡{value}已激活，可在药店使用。",
            "社会保障卡号码：{value}，参保地北京。",
            "请持社保卡{value}到医院就诊。",
            "查询社保缴费记录，卡号{value}。",
            "{name}社保卡{value}遗失，需补办。",
            "社保卡{value}余额查询：医保个人账户500元。",
            "单位为员工{name}办理社保卡，卡号{value}。",
        ]

        for i in range(count):
            ss_card = self.generate_social_security_card()
            context = random.choice(contexts)
            name = random.choice(SURNAMES) + random.choice(['明', '红', '强', '芳', '丽', '伟'])

            text = context.format(value=ss_card, name=name)

            entities = [{"type": "SOCIAL_SECURITY_CARD", "value": ss_card}]
            if '{name}' in context:
                entities.append({"type": "PERSON_NAME", "value": name})

            self.generated_samples.append({
                "input": text,
                "output": {"entities": entities}
            })

        print(f"✅ 社保卡号样本生成完成")

    def generate_mac_address_samples(self, count: int = 200):
        """生成MAC地址样本"""
        print(f"\n🔄 生成MAC地址样本 ({count}个)...")

        contexts = [
            "网卡MAC地址{value}已绑定。",
            "设备MAC地址为{value}，请记录。",
            "路由器MAC：{value}，SSID：HomeWiFi。",
            "DHCP租约显示MAC{value}获取IP{ip}。",
            "MAC地址过滤规则添加{value}。",
            "网络监控：MAC{value}流量异常。",
            "无线网卡MAC地址{value}连接失败。",
            "交换机端口绑定MAC{value}。",
            "设备标识：MAC地址{value}。",
            "ARP表显示{ip}对应MAC{value}。",
        ]

        for i in range(count):
            mac = self.generate_mac_address()
            context = random.choice(contexts)
            ip = self.generate_ip_address()

            text = context.format(value=mac, ip=ip)

            entities = [{"type": "MAC_ADDRESS", "value": mac}]
            if '{ip}' in context:
                entities.append({"type": "IP_ADDRESS", "value": ip})

            self.generated_samples.append({
                "input": text,
                "output": {"entities": entities}
            })

        print(f"✅ MAC地址样本生成完成")

    def generate_all(self):
        """生成所有11种PII的训练数据"""
        print("="*70)
        print("🚀 开始生成11种新PII训练数据")
        print("="*70)
        print(f"目标总量: 14,000样本")
        print(f"输出文件: {self.output_file}")
        print("="*70)

        # 按优化后的数量生成
        self.generate_bank_card_samples(2000)
        self.generate_vehicle_plate_samples(1500)
        self.generate_passport_samples(1500)
        self.generate_hk_macau_pass_samples(1500)
        self.generate_driver_license_samples(1500)
        self.generate_social_security_card_samples(1500)
        self.generate_unified_social_credit_code_samples(1500)
        self.generate_postal_code_samples(800)
        self.generate_ip_address_samples(500)
        self.generate_ipv6_address_samples(500)
        self.generate_mac_address_samples(200)

        # 打乱顺序
        random.shuffle(self.generated_samples)

        # 保存到文件
        print(f"\n💾 保存到文件: {self.output_file}")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for sample in self.generated_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"\n{'='*70}")
        print(f"✅ 数据生成完成！")
        print(f"{'='*70}")
        print(f"总样本数: {len(self.generated_samples)}")
        print(f"文件大小: {len(open(self.output_file).read()) / 1024 / 1024:.2f} MB")

        # 统计各类型数量
        type_counts = {}
        for sample in self.generated_samples:
            for entity in sample["output"]["entities"]:
                pii_type = entity["type"]
                type_counts[pii_type] = type_counts.get(pii_type, 0) + 1

        print(f"\n按PII类型统计:")
        print(f"{'-'*70}")
        for pii_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {pii_type:<30} {count:>6}个")

        print(f"\n{'='*70}")
        print(f"🎉 全部完成！可以开始训练了！")
        print(f"{'='*70}")

if __name__ == "__main__":
    generator = PIIDataGenerator()
    generator.generate_all()
